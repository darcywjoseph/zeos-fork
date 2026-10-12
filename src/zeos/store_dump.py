# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The kernel's span store, written beside the journal.

The journal already determines the store: every span it archives was decoded or
injected first, so the debugger can rebuild the store from the journal alone. This
file is the kernel's own object, written once at the end of a run in the order spans
were first archived, so the two can be compared. Like the machine trace it is kept
apart from the byte-identical record the determinism gate compares.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zeos.core import serde
from zeos.core.store import SpanStore

__all__ = ["write_store", "read_store"]


def write_store(store: SpanStore, path: Path) -> list[dict[str, Any]]:
    """One JSON line per entry: the archived span, and how many stubs point at it."""
    rows = [{**serde.encode(span), "refs": store.refs(span.store_id)} for span in store.spans()]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return rows


def read_store(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
