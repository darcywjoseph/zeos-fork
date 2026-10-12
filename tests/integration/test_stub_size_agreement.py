# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""The eviction planner and the eviction executor must agree on the size of a stub.

``stub_size`` is what the planner subtracts to decide an eviction is worth making;
``render_stub`` around ``summarise`` is what the executor splices in. When the two
disagree, the planner approves evictions that free nothing, and a stub is evicted
into another stub of the same size.
"""

from __future__ import annotations

from tests.integration.test_stage_d_virtual_context import LONG_JOB, build, chatter, of
from zeos.core.events import SegmentEvicted
from zeos.core.ids import DescriptorName
from zeos.core.residency import render_stub, stub_size, summarise
from zeos.machine.base import tokens_from_text


def test_planner_and_executor_agree_on_stub_size() -> None:
    kernel, _ = build([LONG_JOB], {"patrol": chatter(1) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()
    record = max(job.segments.all(), key=lambda r: r.tokens)
    tokens = kernel.machine.transcript(job.job_id)[record.start : record.end]

    planned, budget = stub_size(record.tokens)
    rendered = render_stub(record, summarise(tokens, budget=budget), store_id="a" * 32)
    assert len(tokens_from_text(rendered)) == planned


def test_every_eviction_frees_tokens() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(30) + [{"exit": True}]})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    evictions = of(events, SegmentEvicted)
    assert evictions
    assert all(e.freed_tokens > 0 for e in evictions), [
        (e.segment, e.freed_tokens) for e in evictions if e.freed_tokens <= 0
    ]
