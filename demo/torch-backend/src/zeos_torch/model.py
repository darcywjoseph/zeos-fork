# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Darcy Joseph
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Loaded weights and tokenizer, and where they come from, kept apart from any context."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.cache_utils import DynamicLayer

__all__ = [
    "DEFAULT_REPO",
    "DEFAULT_DIR",
    "DenseAttentionRequired",
    "TorchModel",
    "default_path",
    "fetch",
    "require_dense",
]

DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
#: Where ``fetch`` puts the default weights, under ``models/``: about 1GB of safetensors.
DEFAULT_DIR = "Qwen2.5-0.5B-Instruct"


class DenseAttentionRequired(ValueError):
    """The model has a layer that is not full softmax attention over a truncatable cache.

    Sliding-window and linear-attention layers have no attention row over the whole
    context to mask or measure, and their caches cannot be cut back to an arbitrary
    position, so a backend over them could honour neither the mask nor ``trunc``.
    """


def default_path(root: Path | None = None) -> Path:
    env = os.environ.get("ZEOS_TORCH_MODEL")
    if env:
        return Path(env)
    return (root or Path.cwd()) / "models" / DEFAULT_DIR


def fetch(repo: str = DEFAULT_REPO, *, into: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

    target = into or (Path.cwd() / "models" / repo.rsplit("/", 1)[-1])
    target.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo,
            local_dir=str(target),
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
    )


def require_dense(config: PretrainedConfig, name: str) -> None:
    """Raise ``DenseAttentionRequired`` if the config declares any layer that is not full
    attention, or builds any cache layer other than a plain dynamic one."""
    layer_types: list[str] = getattr(config, "layer_types", None) or []
    other = sorted({t for t in layer_types if t != "full_attention"})
    if other:
        raise DenseAttentionRequired(f"{name}: has {other} layers")
    kinds = sorted(
        {type(layer).__name__ for layer in DynamicCache(config=config).layers}
        - {DynamicLayer.__name__}
    )
    if kinds:
        raise DenseAttentionRequired(f"{name}: its cache has {kinds} layers")


class TorchModel:
    """A causal LM loaded for measurement: eager attention, float32, on one device."""

    def __init__(self, path: str | Path, *, device: str = "cpu") -> None:
        self.path = str(path)
        self.device = torch.device(device)
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(self.path),  # pyright: ignore[reportUnknownMemberType]
        )
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            self.path,
            # Eager is the only implementation that materialises the attention weights.
            attn_implementation="eager",
            dtype=torch.float32,
        )
        # The base ``Module`` methods, which act in place: transformers wraps its own in
        # decorators the type checkers cannot see through.
        torch.nn.Module.to(model, self.device)
        torch.nn.Module.eval(model)
        self.model = model
        self.config = self.model.config
        require_dense(self.config, self.path)
        #: Rows of the embedding past the tokenizer's vocabulary are padding, not tokens.
        self.n_tokens = len(self.tokenizer)
        self.n_logits = int(self.config.get_text_config().vocab_size)  # pyright: ignore[reportUnknownMemberType]
        self._pieces: dict[int, str] = {}

    def encode(self, text: str, *, first: bool, special: bool = False) -> list[int]:
        """Turn text into ids. Only ``special`` text may spell a special token: anything
        else that looks like ``<|im_start|>`` is tokenized as the characters it is."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=first,
            split_special_tokens=not special,
        )
        return list(cast(list[int], encoded["input_ids"]))

    def piece(self, token: int) -> str:
        cached = self._pieces.get(token)
        if cached is not None:
            return cached
        text = cast(str, self.tokenizer.decode([token]))  # pyright: ignore[reportUnknownMemberType]
        self._pieces[token] = text
        return text

    def special_ids(self) -> frozenset[int]:
        return frozenset(self.tokenizer.all_special_ids)

    def end_ids(self) -> frozenset[int]:
        """Ids that end generation: the tokenizer's EOS and the generation config's."""
        ends: set[int] = set()
        for declared in (
            cast(object, self.tokenizer.eos_token_id),
            cast(object, self.model.generation_config.eos_token_id),  # pyright: ignore[reportUnknownMemberType]
        ):
            if isinstance(declared, int):
                ends.add(declared)
            elif isinstance(declared, list):
                ends.update(cast(list[int], declared))
        if not ends:
            raise ValueError(f"{self.path}: no end-of-generation id in tokenizer or config")
        return frozenset(ends)
