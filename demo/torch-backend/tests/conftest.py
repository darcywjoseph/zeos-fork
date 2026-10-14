# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Darcy Joseph
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures: one model per session, skipped when the weights are absent."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from zeos_torch.machine import TorchMachine
from zeos_torch.model import TorchModel, default_path

ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    # A backend whose bits depend on kernel selection could not hold a journal byte for byte.
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def torch_model() -> TorchModel:
    path = default_path(ROOT)
    if not (path / "config.json").is_file():
        pytest.skip(f"no model at {path}; run 'zeos-torch fetch-model'")
    return TorchModel(path)


@pytest.fixture
def machines() -> Iterator[list[TorchMachine]]:
    """Tracks every machine a test builds so they are all closed at the end."""
    built: list[TorchMachine] = []
    yield built
    for m in built:
        m.close()
