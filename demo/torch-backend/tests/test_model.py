# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Darcy Joseph
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The loader's refusal of models whose attention cannot be masked or truncated."""

from __future__ import annotations

import pytest
from transformers import LlamaConfig, MistralConfig, Qwen2Config, Qwen3NextConfig

from zeos_torch.model import DenseAttentionRequired, require_dense


def test_full_attention_in_every_layer_is_accepted() -> None:
    require_dense(Qwen2Config(num_hidden_layers=2), "qwen2")
    # No layer_types at all is no indication of anything else; the cache check stands.
    require_dense(LlamaConfig(num_hidden_layers=2), "llama")


def test_declared_linear_attention_layers_are_refused() -> None:
    with pytest.raises(DenseAttentionRequired, match="linear_attention"):
        require_dense(Qwen3NextConfig(num_hidden_layers=4), "qwen3-next")


def test_declared_sliding_window_layers_are_refused() -> None:
    config = Qwen2Config(
        num_hidden_layers=2, use_sliding_window=True, sliding_window=8, max_window_layers=0
    )
    with pytest.raises(DenseAttentionRequired, match="sliding_attention"):
        require_dense(config, "qwen2-sliding")


def test_a_sliding_window_cache_is_refused_even_when_no_layer_type_says_so() -> None:
    with pytest.raises(DenseAttentionRequired, match="DynamicSlidingWindowLayer"):
        require_dense(MistralConfig(num_hidden_layers=2, sliding_window=16), "mistral")
