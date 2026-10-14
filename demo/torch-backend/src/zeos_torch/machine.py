# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Darcy Joseph
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A machine backend that runs a Hugging Face forward pass for each decode step, owning
the token sequence and its KV while the kernel owns segments, rings and integrity."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from transformers import DynamicCache
from zeos.core.ids import JobId, TokenKind
from zeos.machine.abi import DEFAULT, SyscallABI
from zeos.machine.base import (
    AttentionHint,
    ContextStats,
    ControlTokenViolation,
    DecodeResult,
    MachineRequest,
    MaskViolation,
    OpKind,
    RawWindow,
    RawWord,
    SpliceResult,
    Token,
)
from zeos.machine.scripted import PAD_TOKEN
from zeos.machine.seat import SyscallParser, SyscallSeat

from zeos_torch.context import Context
from zeos_torch.model import TorchModel

__all__ = ["TorchMachine", "DEFAULT_BLOCK_SIZE"]

DEFAULT_BLOCK_SIZE = 16


class TorchMachine(SyscallSeat):
    """A machine backend with one KV cache per job, whose decode samples one token."""

    def __init__(
        self,
        model: TorchModel,
        *,
        abi: SyscallABI = DEFAULT,
        descriptors: Mapping[str, Sequence[str]] | None = None,
        valued: Mapping[str, Sequence[str]] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        n_ctx: int = 8192,
        n_batch: int = 512,
        chat_template: str | None = "chatml",
        on_command: Callable[[JobId, str, MachineRequest], None] | None = None,
        on_arrival: Callable[[JobId, str], None] | None = None,
    ) -> None:
        super().__init__(
            abi=abi,
            descriptors=descriptors,
            valued=valued,
            on_command=on_command,
            on_arrival=on_arrival,
        )
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if chat_template not in (None, "chatml"):
            raise ValueError(f"unknown chat_template {chat_template!r}")
        self._model = model
        self._block_size = block_size
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._chat_template = chat_template
        self._contexts: dict[JobId, Context] = {}

        self._pad_id = self._choose_pad(model)
        self._ends = model.end_ids()
        #: Ids the sampler may select only when the kernel enables control tokens: block
        #: padding, and the special tokens that spell chat framing.
        self._reserved = frozenset({self._pad_id}) | (model.special_ids() - self._ends)
        reserved = torch.zeros(model.n_logits, dtype=torch.bool)
        reserved[sorted(self._reserved)] = True
        #: Logit rows past the tokenizer's vocabulary, which no id can be sampled from.
        unassigned = torch.zeros(model.n_logits, dtype=torch.bool)
        unassigned[model.n_tokens :] = True
        self._closed = unassigned | reserved
        self._unassigned = unassigned

    @staticmethod
    def _choose_pad(model: TorchModel) -> int:
        """Pick the id block padding is made of, preferring a newline because it reads as text."""
        newline = model.encode("\n", first=False)
        if len(newline) == 1:
            return newline[0]
        pad = model.tokenizer.pad_token_id  # pyright: ignore[reportUnknownMemberType]
        if isinstance(pad, int):
            return pad
        return min(model.end_ids())

    # -- lifecycle ----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    #: The chat turn markers, the only text encoded with special tokens recognised.
    _CHATML_OPEN = "<|im_start|>user\n"
    _CHATML_TURN = "<|im_end|>\n<|im_start|>assistant\n"
    _CHATML_ARRIVE = "<|im_end|>\n<|im_start|>user\n"

    def _frame_into(self, ctx: Context, text: str, index: int, *, before: bool) -> None:
        """Fold chat framing into the span of the token at ``index``, so no kernel offset moves."""
        ids = self._model.encode(text, first=False, special=True)
        if ids:
            ctx.drop_kv(ctx.frame(index, ids, before=before))

    def create_context(self, job: JobId, descriptor: str = "") -> None:
        if job in self._contexts:
            self.destroy_context(job)
        self._contexts[job] = Context(
            descriptor=descriptor,
            parser=SyscallParser(self.abi),
            cache=DynamicCache(config=self._model.config),
            tags=("descriptor",),
        )

    def destroy_context(self, job: JobId) -> None:
        self._contexts.pop(job, None)

    def _ctx_of(self, job: JobId) -> Context:
        ctx = self._contexts.get(job)
        if ctx is None:
            raise KeyError(f"no context for job {job}; create_context first")
        return ctx

    def stats(self, job: JobId) -> ContextStats:
        ctx = self._ctx_of(job)
        n = len(ctx.tokens)
        return ContextStats(
            resident_tokens=n,
            blocks=self._block_count(n),
            open_segment_tokens=n % self._block_size,
        )

    def _block_count(self, n: int) -> int:
        return (n + self._block_size - 1) // self._block_size

    # -- the forward pass ----------------------------------------------------

    @torch.no_grad()
    def _flush(self, ctx: Context) -> None:
        """Run forward passes over every id not yet in the KV, keeping the last logits."""
        pending = ctx.ids[ctx.n_in_kv :]
        if not pending:
            return
        if len(ctx.ids) > self._n_ctx:
            raise RuntimeError(
                f"job context is {len(ctx.ids)} model tokens, over the {self._n_ctx} this "
                f"machine allots each job. Raise n_ctx, or lower the descriptor's "
                f"context.window so the kernel evicts before the model runs out."
            )
        device = self._model.device
        for offset in range(0, len(pending), self._n_batch):
            chunk = pending[offset : offset + self._n_batch]
            first = ctx.n_in_kv
            out = self._model.model(
                input_ids=torch.tensor([chunk], device=device),
                position_ids=torch.arange(first, first + len(chunk), device=device)[None],
                past_key_values=ctx.cache,
                use_cache=True,
                logits_to_keep=1,
            )
            ctx.logits = out.logits[0, -1].to("cpu")

    def _next_logits(self, ctx: Context) -> torch.Tensor:
        """The logits for the position after the last id, redoing one id if need be."""
        if not ctx.ids:
            raise RuntimeError(
                "cannot decode an empty context; the kernel injects the descriptor body "
                "before the first decode"
            )
        if ctx.n_in_kv == len(ctx.ids) and ctx.logits is None:
            ctx.drop_kv(len(ctx.ids) - 1)
        self._flush(ctx)
        assert ctx.logits is not None
        return ctx.logits

    def _encode(
        self, tokens: Sequence[Token], *, first: bool
    ) -> tuple[list[Token], list[int], list[int]]:
        """Turn kernel tokens into ids and spans, one word at a time."""
        out_tokens: list[Token] = []
        out_ids: list[int] = []
        out_spans: list[int] = []
        for k, tok in enumerate(tokens):
            opening = first and k == 0
            # A leading space marks a word boundary, which the very first token lacks.
            ids = self._model.encode(tok.text if opening else " " + tok.text, first=opening)
            if not ids:
                ids = [self._pad_id]
            out_tokens.append(tok)
            out_ids.extend(ids)
            out_spans.append(len(ids))
        return out_tokens, out_ids, out_spans

    # -- the five ops --------------------------------------------------------

    def decode(self, job: JobId, *, allow_control: bool) -> DecodeResult:
        ctx = self._ctx_of(job)
        if self._chat_template == "chatml" and not ctx.turn_open and ctx.tokens:
            # Everything injected so far was the prompt, so close that turn and open the
            # model's own.
            ctx.turn_index = len(ctx.tokens) - 1
            self._frame_into(ctx, self._CHATML_TURN, ctx.turn_index, before=False)
            ctx.turn_open = True

        scores = self._next_logits(ctx).clone()
        # Reserved ids are made unselectable before sampling, never filtered after (AM §9).
        scores[self._closed if not allow_control else self._unassigned] = float("-inf")
        tid = int(torch.argmax(scores))

        if tid in self._ends:
            # End of generation is an EXIT request the kernel checks, not an exception.
            return DecodeResult(
                tokens=(),
                request=MachineRequest(op=OpKind.EXIT),
                attention=None,
                attention_hint=AttentionHint(tags=ctx.tags),
            )
        if tid in self._reserved and not allow_control:
            raise ControlTokenViolation(
                f"job {job} sampled reserved id {tid} while control tokens were disabled"
            )

        piece = self._model.piece(tid)
        token = Token(piece, TokenKind.CONTROL if tid in self._reserved else TokenKind.NORMAL)
        ctx.spoken = True

        before = len(ctx.tokens)
        ctx.extend([token], [tid], [1])
        after = len(ctx.tokens)

        return DecodeResult(
            tokens=(token,),
            request=self.consume(job, ctx.parser, piece),
            attention=None,
            attention_hint=AttentionHint(tags=ctx.tags),
            at_block_boundary=after % self._block_size == 0 and after != before,
        )

    def inject(self, job: JobId, tokens: Sequence[Token]) -> tuple[int, int]:
        ctx = self._ctx_of(job)
        start = len(ctx.tokens)
        first = not ctx.tokens
        new_tokens, new_ids, new_spans = self._encode(tokens, first=not ctx.ids)
        ctx.extend(new_tokens, new_ids, new_spans)
        # An arrival lands mid-flight when the model's turn is open; anything before that
        # is still the prompt being put together.
        if new_tokens and self._chat_template == "chatml":
            if first:
                self._frame_into(ctx, self._CHATML_OPEN, start, before=True)
            elif ctx.turn_open:
                # Close the job's turn and let the sender speak; ``decode`` reopens the
                # job's turn later, so a run of arrivals shares one turn.
                self._frame_into(ctx, self._CHATML_ARRIVE, start, before=True)
                ctx.turn_open = False
                ctx.turn_index = None
        # Reporting keys on ``spoken``, not on the framing above, because a run of arrivals
        # shares one turn but each one still arrived.
        if ctx.spoken and new_tokens:
            self.note_arrival(job, " ".join(t.text for t in new_tokens))
        # New content is the most likely thing the next step attends to.
        if new_tokens:
            ctx.tags = ("self",)
        return start, len(ctx.tokens)

    def trunc(self, job: JobId, at: int) -> int:
        ctx = self._ctx_of(job)
        if at < 0 or at > len(ctx.tokens):
            raise IndexError(f"trunc at {at} outside [0, {len(ctx.tokens)}]")
        dropped = len(ctx.tokens) - at
        if dropped == 0:
            return 0
        ctx.drop_kv(ctx.kv_offset(at))
        ctx.truncate(at)
        if ctx.turn_index is not None and at <= ctx.turn_index:
            ctx.turn_index = None
            ctx.turn_open = False
        return dropped

    def fork(self, parent: JobId, child: JobId) -> int:
        src = self._ctx_of(parent)
        if child not in self._contexts:
            self.create_context(child, src.descriptor)
        self._ctx_of(child).copy_from(src)
        return len(src.tokens)

    def splice(self, job: JobId, start: int, end: int, tokens: Sequence[Token]) -> SpliceResult:
        """Replace a range of tokens, the one operation that moves the offsets after it."""
        ctx = self._ctx_of(job)
        if not 0 <= start <= end <= len(ctx.tokens):
            raise IndexError(f"splice [{start}, {end}) outside context of {len(ctx.tokens)}")
        downstream = len(ctx.tokens) - end
        kv_start = ctx.kv_offset(start)
        new_tokens, new_ids, new_spans = self._encode(tokens, first=kv_start == 0)

        ctx.drop_kv(kv_start)
        ctx.replace(start, end, new_tokens, new_ids, new_spans)
        if ctx.turn_index is not None and ctx.turn_index >= start:
            if ctx.turn_index < end:
                # The word carrying the turn marker was replaced, and the marker with it.
                ctx.turn_index = None
                ctx.turn_open = False
            else:
                ctx.turn_index += len(new_tokens) - (end - start)
        return SpliceResult(tokens_in=len(new_tokens), invalidated_downstream=downstream)

    # -- the serving-stack contract ------------------------------------------

    def set_mask(self, job: JobId, allowed_blocks: frozenset[int]) -> None:
        """Install the allowed-block bitmap, refusing one that narrows what the job can see."""
        ctx = self._ctx_of(job)
        live = frozenset(range(self._block_count(len(ctx.tokens))))
        if not live <= allowed_blocks:
            raise MaskViolation(
                f"job {job}: this backend does not enforce an allowed-block bitmap, so it "
                f"refuses one that narrows {sorted(live)} to {sorted(live & allowed_blocks)}. "
                f"See ZEOS-AM §8.1."
            )
        ctx.mask = allowed_blocks

    def visible_blocks(self, job: JobId) -> frozenset[int]:
        ctx = self._ctx_of(job)
        # No mask means every block, which is not the same as a mask allowing none.
        every = frozenset(range(self._block_count(len(ctx.tokens))))
        return every if ctx.mask is None else every & ctx.mask

    def pad_to_block(self, job: JobId) -> int:
        ctx = self._ctx_of(job)
        remainder = len(ctx.tokens) % self._block_size
        if remainder == 0:
            return 0
        padding = self._block_size - remainder
        ctx.extend([PAD_TOKEN] * padding, [self._pad_id] * padding, [1] * padding)
        return padding

    def blocks_for_range(self, job: JobId, start: int, end: int) -> frozenset[int]:
        self._ctx_of(job)
        if end <= start:
            return frozenset()
        return frozenset(range(start // self._block_size, (end - 1) // self._block_size + 1))

    def transcript(self, job: JobId) -> tuple[Token, ...]:
        return tuple(self._ctx_of(job).tokens)

    def raw(self, job: JobId) -> RawWindow:
        """Each word's pieces and the framing around them; framing after a word is
        reported ahead of the next, and after the last word as ``trailing``."""
        ctx = self._ctx_of(job)
        piece = self._model.piece
        words: list[RawWord] = []
        carried: list[str] = []
        at = 0
        for span, (head, tail) in zip(ctx.spans, ctx.framing, strict=True):
            ids = ctx.ids[at : at + span]
            at += span
            own = span - head - tail
            words.append(
                RawWord(
                    pieces=tuple(piece(t) for t in ids[head : head + own]),
                    framing=tuple(carried + [piece(t) for t in ids[:head]]),
                )
            )
            carried = [piece(t) for t in ids[head + own :]]
        return RawWindow(words=tuple(words), kv_resident=ctx.n_in_kv, trailing=tuple(carried))

    # -- outside the Protocol -------------------------------------------------

    def lines(self, job: JobId) -> tuple[str, ...]:
        """The syscall lines this job has produced, for display only."""
        return tuple(self._ctx_of(job).parser.lines)

    def close(self) -> None:
        self._contexts.clear()
