# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Darcy Joseph
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Per-job state: the kernel's words, the model ids behind them, and the KV over those ids."""

from __future__ import annotations

import copy
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache
from zeos.machine.base import Token
from zeos.machine.seat import SyscallParser

__all__ = ["Context"]


@dataclass
class Context:
    """One job's context. Every change to ``ids`` goes through a method here, so that
    ``owner`` -- each model position's kernel offset -- never drifts from ``spans``."""

    descriptor: str
    parser: SyscallParser
    cache: DynamicCache
    #: The kernel-visible token sequence T.
    tokens: list[Token] = field(default_factory=list[Token])
    #: The model ids backing T, flat, chat framing included.
    ids: list[int] = field(default_factory=list[int])
    #: ``spans[i]`` is how many ids element ``i`` of T occupies, framing included.
    spans: list[int] = field(default_factory=list[int])
    #: ``framing[i]`` is how many of those ids are chat framing, before and after the word.
    framing: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    #: ``owner[p]`` is the kernel offset model position ``p`` belongs to. Framing ids
    #: belong to the word they are folded into, so their attention lands in its block.
    owner: list[int] = field(default_factory=list[int])
    #: Next-token logits after the last id in the KV, or None if that position has not
    #: been the last one through a forward pass since the KV last changed.
    logits: torch.Tensor | None = None
    mask: frozenset[int] | None = None
    #: Tags for the attention hint, naming what this step most likely attends to.
    tags: tuple[str, ...] = ()
    #: Whether the model's own turn has been opened.
    turn_open: bool = False
    #: Which element of T carries the turn marker in its span, so truncating past it
    #: reopens the prompt turn.
    turn_index: int | None = None
    #: Whether the job has decoded anything yet.
    spoken: bool = False

    @property
    def n_in_kv(self) -> int:
        """How many ids have a forward pass behind them; the rest are pending."""
        return self.cache.get_seq_length()

    def kv_offset(self, at: int) -> int:
        """Turn a kernel offset into a model position, the one place the two units meet."""
        return bisect_left(self.owner, at)

    def extend(self, tokens: Sequence[Token], ids: Sequence[int], spans: Sequence[int]) -> None:
        start = len(self.tokens)
        self.tokens.extend(tokens)
        self.ids.extend(ids)
        self.spans.extend(spans)
        self.framing.extend([(0, 0)] * len(spans))
        for i, span in enumerate(spans):
            self.owner.extend([start + i] * span)

    def frame(self, index: int, ids: Sequence[int], *, before: bool) -> int:
        """Fold framing ids into the span of word ``index``. Returns the position they
        went in at, which the caller must cut the KV back to if it already covers it."""
        at = self.kv_offset(index) if before else self.kv_offset(index + 1)
        self.ids[at:at] = ids
        self.owner[at:at] = [index] * len(ids)
        self.spans[index] += len(ids)
        head, tail = self.framing[index]
        self.framing[index] = (head + len(ids), tail) if before else (head, tail + len(ids))
        return at

    def truncate(self, at: int) -> None:
        """Drop words from ``at`` on, with their ids. The KV is the caller's to cut."""
        kv_at = self.kv_offset(at)
        del self.tokens[at:]
        del self.ids[kv_at:]
        del self.spans[at:]
        del self.framing[at:]
        del self.owner[kv_at:]

    def replace(
        self,
        start: int,
        end: int,
        tokens: Sequence[Token],
        ids: Sequence[int],
        spans: Sequence[int],
    ) -> None:
        """Replace words ``[start, end)``, renumbering every position downstream."""
        kv_start = self.kv_offset(start)
        kv_end = self.kv_offset(end)
        self.tokens[start:end] = tokens
        self.ids[kv_start:kv_end] = ids
        self.spans[start:end] = spans
        self.framing[start:end] = [(0, 0)] * len(spans)
        del self.owner[kv_start:]
        for i, span in enumerate(self.spans[start:], start=start):
            self.owner.extend([i] * span)

    def drop_kv(self, position: int) -> None:
        """Discard the KV from model position ``position`` on."""
        excess = self.n_in_kv - position
        if excess > 0:
            # A negative argument counts positions to remove; zero or positive means
            # something else in this transformers line, so it is never passed.
            self.cache.crop(-excess)
            self.logits = None

    def copy_from(self, src: Context) -> None:
        """Take the parent's tokens, KV, turn state and mask, keeping this context's own
        descriptor and parser. The KV is copied outright rather than shared."""
        self.tokens = list(src.tokens)
        self.ids = list(src.ids)
        self.spans = list(src.spans)
        self.framing = list(src.framing)
        self.owner = list(src.owner)
        self.cache = copy.deepcopy(src.cache)
        self.logits = None if src.logits is None else src.logits.clone()
        self.turn_open = src.turn_open
        self.turn_index = src.turn_index
        self.spoken = src.spoken
        # A child starts from the parent's visibility and can only lose ground from there.
        self.mask = src.mask
