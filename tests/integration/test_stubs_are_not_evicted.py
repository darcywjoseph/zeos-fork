# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A stub is never an eviction candidate.

Its content is already in the store and the way back is a fault on the segment it
replaced. Evicting it would archive its own framing and summarise a summary. Under
pressure the planner says so, the same way it declines any other span.
"""

from __future__ import annotations

from tests.integration.test_stage_d_virtual_context import LONG_JOB, build, chatter, of
from zeos.core.events import SegmentEvicted
from zeos.core.ids import DescriptorName
from zeos.core.residency import ContextPolicy, plan_eviction


def test_no_eviction_replaces_a_stub() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(30) + [{"exit": True}]})
    kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()

    evictions = of(events, SegmentEvicted)
    assert evictions
    stubs = {e.stub for e in evictions}
    assert not [e.segment for e in evictions if e.segment in stubs]


def test_a_stub_is_skipped_with_a_reason_naming_its_original() -> None:
    kernel, events = build([LONG_JOB], {"patrol": chatter(30) + [{"exit": True}]})
    job = kernel.spawn(DescriptorName("patrol"))
    kernel.run_until_quiescent()
    originals = {e.stub: e.segment for e in of(events, SegmentEvicted)}
    assert originals

    for record in job.segments.all():
        record.attn.ema = 0.0  # everything is cold: any span would go
    plan = plan_eviction(
        job.segments,
        ContextPolicy(window=100, high_watermark=0.1, low_watermark=0.05),
        resident_tokens=1000,
        current_token_clock=10_000,
    )
    assert not [c for c in plan.candidates if c.record.id in originals]
    skipped = {c.record.id: c.reason for c in plan.skipped if c.record.id in originals}
    assert skipped
    for stub, reason in skipped.items():
        assert "stub" in reason and str(originals[stub]) in reason, reason
