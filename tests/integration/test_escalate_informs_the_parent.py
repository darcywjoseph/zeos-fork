# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""A child that faults under ``on_fault: escalate`` informs its parent."""

from __future__ import annotations

from typing import Any

from zeos.core.events import Event, FaultDispatched, FaultEscalated, Injected, JobBlocked
from zeos.core.ids import DescriptorName, FaultKind, JobState, PipeName
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.scripted import Script, ScriptedMachine
from zeos.world.store import WorldStore

RESULT = PipeName("child.result")

# The child is less urgent than its parent, so the parent is parked on the child's pipe
# by the time the child faults.
DESCRIPTORS: list[dict[str, Any]] = [
    {"name": "parent", "priority": 50, "children": ["child"], "pipes": {"stdin": str(RESULT)}},
    {"name": "child", "priority": 60, "pipes": {"stdout": str(RESULT)}},
]
CHILD = [{"write": {"pipe": "not.bound", "text": "x"}}, {"exit": True}]


def _run(first: str = "parent", waits: dict[str, Any] | None = None) -> tuple[Kernel, list[Event]]:
    parent = [{"spawn": "child"}, waits or {"read": str(RESULT)}, {"exit": True}]
    scripts = {"parent": parent, "child": CHILD}
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(d["name"])): Descriptor.from_frontmatter(d) for d in DESCRIPTORS
        },
        machine=ScriptedMachine({n: Script.from_spec(s) for n, s in scripts.items()}, block_size=8),
        pipes=PipeTable([PipeSpec(name=RESULT)]),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=events,
        config=KernelConfig(case="escalate", max_ticks=200),
    )
    kernel.start()
    kernel.spawn(DescriptorName(first))
    kernel.run_until_quiescent()
    return kernel, events


def _job(kernel: Kernel, name: str) -> Any:
    return next(j for j in kernel.sched.jobs() if j.name == DescriptorName(name))


def _after_the_fault(events: list[Event]) -> list[Event]:
    return events[next(i for i, e in enumerate(events) if isinstance(e, FaultDispatched)) :]


def _kernel_notices(events: list[Event], job: Any) -> list[str]:
    return [
        " ".join(e.text)
        for e in events
        if isinstance(e, Injected) and e.job == job and str(e.pipe) == "kernel"
    ]


def test_the_child_faults_under_escalate_while_the_parent_waits_on_it() -> None:
    kernel, events = _run()
    parent, child = _job(kernel, "parent"), _job(kernel, "child")
    assert child.state is JobState.FAULTED
    assert [(e.job, e.policy) for e in events if isinstance(e, FaultDispatched)] == [
        (child.job_id, "escalate")
    ]
    before = events[: events.index(_after_the_fault(events)[0])]
    assert [(e.job, e.pipe) for e in before if isinstance(e, JobBlocked)] == [
        (parent.job_id, RESULT)
    ]


def test_escalate_informs_the_parent_and_wakes_it() -> None:
    kernel, events = _run()
    parent, child = _job(kernel, "parent"), _job(kernel, "child")
    escalated = [e for e in events if isinstance(e, FaultEscalated)]
    assert [(e.job, e.child, e.fault) for e in escalated] == [
        (parent.job_id, child.job_id, FaultKind.CAPABILITY)
    ]
    assert _kernel_notices(_after_the_fault(events), parent.job_id)
    assert parent.state is JobState.DONE, "the parent still waits on the child's pipe"


def test_a_parent_parked_on_a_select_is_woken_too() -> None:
    kernel, events = _run(waits={"select": [str(RESULT)]})
    assert any(isinstance(e, JobBlocked) and e.reason == "select" for e in events)
    assert _job(kernel, "parent").state is JobState.DONE


def test_the_notice_names_the_child_and_never_its_detail() -> None:
    kernel, events = _run()
    [notice] = _kernel_notices(_after_the_fault(events), _job(kernel, "parent").job_id)
    assert "kind=capability_fault" in notice and "child=child>" in notice
    assert "not.bound" not in notice, "the child's own words reached the parent"


def test_a_job_with_no_parent_faults_as_under_abort() -> None:
    kernel, events = _run(first="child")
    assert _job(kernel, "child").state is JobState.FAULTED
    assert [e for e in events if isinstance(e, FaultEscalated)] == []
