# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""What resets a job's starvation count (core §5.5), and what does not."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from zeos.core.embodiment import PlatformProfile
from zeos.core.events import (
    Decoded,
    Disembodied,
    Event,
    FaultRaised,
    FrameSent,
    GateAnswered,
    JobBlocked,
    JobPreempted,
    JobSpawned,
    PipeReadEvent,
    PipeWritten,
    ResourceAcquired,
    ResourceBlocked,
    ResourceReleased,
)
from zeos.core.gates import GateSpec, GateTable
from zeos.core.ids import (
    DescriptorName,
    FaultKind,
    JobState,
    PipeName,
    Principal,
    Priority,
    ResourceKind,
    ResourceName,
    VectorName,
)
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pcb import Job
from zeos.core.pipes import PipeSpec, PipeTable
from zeos.core.resources import ResourceSpec, ResourceTable
from zeos.core.topology import LinkSpec
from zeos.core.vectors import VectorSpec, VectorTable
from zeos.descriptor.schema import Descriptor
from zeos.machine.base import MachineRequest, OpKind
from zeos.machine.scripted import Script, ScriptedMachine, Step
from zeos.transport.link import LinkTransport
from zeos.world.store import WorldStore

K = KernelConfig().starvation_limit
IRQ = PipeName("irq")
OUT = PipeName("out")
SINK = PipeName("out.sink")
DOOR = PipeName("act.door")
INBOX = PipeName("inbox")
UPLINK = PipeName("uplink")
GATE_REQ = PipeName("gate.req")
GATE_VERDICT = PipeName("gate.verdict")
RIVAL_IN = PipeName("rival.in")
LOCKS = 40
STEPS = 200
CARRIER = PlatformProfile(
    name="carrier-7",
    locomotion="wheeled",
    tooling=frozenset({"gripper-std"}),
    battery=0.95,
    location="bay-3",
)
BODY: dict[str, Any] = {"tooling": ["gripper-std"]}

WORK: dict[str, Any] = {"emit": "work"}
GATED: dict[str, Any] = {"write": {"pipe": "tools", "text": "open"}}

Did = Callable[[Event, Job], bool]
Between = Callable[[Kernel, Job, list[Event]], None]


def _write(alias: str) -> dict[str, Any]:
    return {"write": {"pipe": alias, "text": "move"}}


def _script(steps: Sequence[Mapping[str, Any]]) -> Script:
    return Script.from_spec([*steps, {"exit": True}])


def _run(
    victim_script: Script,
    *,
    stepped: Did,
    on_fault: str | None = None,
    guard: str | None = None,
    link: bool = False,
    feed: bool = False,
    body: bool = False,
    between: Between | None = None,
) -> tuple[Job, list[Event]]:
    """Interrupt the victim each time it is running and has taken its step since the
    last interruption, until it has been interrupted K+2 times or has stopped."""
    victim: dict[str, Any] = {
        "name": "victim",
        "priority": 80,
        "pipes": {
            "stdout": str(OUT),
            "stdin": str(INBOX),
            "tools": str(DOOR),
            "sink": str(SINK),
            "uplink": str(UPLINK),
        },
        "children": ["child"],
    }
    if on_fault is not None:
        victim["on_fault"] = on_fault
    fronts: list[dict[str, Any]] = [
        victim,
        {"name": "handler", "priority": 5, "budget": {"tokens": 64}, "pipes": {"stdin": "irq"}},
        {"name": "guard", "priority": 10, "pipes": {"stdin": "gate.req", "stdout": "gate.verdict"}},
        {"name": "child", "priority": 90},
    ]
    scripts = {
        "victim": victim_script,
        "handler": _script([WORK]),
        "guard": _script(
            [{"read": "stdin"}, {"write": {"pipe": "stdout", "text": guard or "veto"}}]
        ),
        "child": _script([]),
    }
    if body:
        victim["requires"] = BODY
        fronts.append(
            {"name": "rival", "priority": 50, "requires": BODY, "pipes": {"stdin": str(RIVAL_IN)}}
        )
        scripts["rival"] = _script([{"read": "stdin"}, WORK])
    gates = GateTable(
        []
        if guard is None
        else [
            GateSpec(
                pipe=DOOR,
                descriptor=DescriptorName("guard"),
                requests=GATE_REQ,
                verdicts=GATE_VERDICT,
                timeout_ticks=100_000,
            )
        ]
    )
    events: list[Event] = []
    kernel = Kernel(
        descriptors={
            DescriptorName(str(f["name"])): Descriptor.from_frontmatter(f, body="b") for f in fronts
        },
        machine=ScriptedMachine(scripts, block_size=16),
        pipes=PipeTable(
            [
                PipeSpec(name=IRQ, device=True, capacity_tokens=64),
                PipeSpec(name=OUT, capacity_tokens=100_000),
                PipeSpec(name=SINK, sink=True, capacity_tokens=100_000),
                PipeSpec(name=DOOR, principal=Principal.DEVICE, world_object="door"),
                PipeSpec(name=INBOX, device=True, capacity_tokens=64),
                PipeSpec(name=UPLINK),
                PipeSpec(name=GATE_REQ),
                PipeSpec(name=GATE_VERDICT),
                PipeSpec(name=RIVAL_IN, device=True, capacity_tokens=64),
            ]
        ),
        vectors=VectorTable(
            [
                VectorSpec(
                    name=VectorName("irq"),
                    source=IRQ,
                    handler=DescriptorName("handler"),
                    priority=Priority(5),
                )
            ]
        ),
        world=WorldStore(),
        resources=ResourceTable(
            [ResourceSpec(name=ResourceName(f"lock.{i}")) for i in range(LOCKS)]
        ),
        gates=gates,
        link=LinkTransport(LinkSpec(), pipes=frozenset({UPLINK})) if link else None,
        platforms=(CARRIER,) if body else (),
        journal_sink=events,
        config=KernelConfig(case="starvation-progress"),
    )
    kernel.start()
    job = kernel.spawn(DescriptorName("victim"))
    since = 0
    for _ in range(2000):
        if job.state.is_terminal or len(_windows(events, job)) > K:
            break
        if feed:
            kernel.deliver(INBOX, "data")
        if job.state is JobState.RUNNING and any(stepped(e, job) for e in events[since:]):
            kernel.deliver(IRQ, "ping")
            since = len(events)
        if between is not None:
            between(kernel, job, events)
        kernel.tick()
    return job, events


def _windows(events: Sequence[Event], victim: Job) -> list[Sequence[Event]]:
    """The journal between each pair of consecutive preemptions of the victim."""
    marks = [
        i for i, e in enumerate(events) if isinstance(e, JobPreempted) and e.job == victim.job_id
    ]
    return [events[a:b] for a, b in zip(marks, marks[1:], strict=False)]


def _starvation(events: Sequence[Event], victim: Job) -> list[FaultRaised]:
    return [
        e
        for e in events
        if isinstance(e, FaultRaised) and e.job == victim.job_id and e.fault is FaultKind.STARVATION
    ]


def _preemptions_before_starvation(events: Sequence[Event], victim: Job) -> int | None:
    count = 0
    for e in events:
        if isinstance(e, JobPreempted) and e.job == victim.job_id:
            count += 1
        elif (
            isinstance(e, FaultRaised)
            and e.job == victim.job_id
            and e.fault is FaultKind.STARVATION
        ):
            return count
    return None


def _progress(e: Event, victim: Job) -> bool:
    """An outcome by the victim that counts as progress. The guard's request is written
    on the victim's behalf and is not one."""
    me = victim.job_id
    return (
        (isinstance(e, PipeWritten) and e.job == me and e.pipe != GATE_REQ)
        or (isinstance(e, PipeReadEvent) and e.job == me)
        or (isinstance(e, ResourceAcquired | ResourceReleased) and e.job == me)
        or (isinstance(e, JobSpawned) and e.parent == me)
        or isinstance(e, FrameSent)
    )


def _assert_never_starves(victim: Job, events: Sequence[Event], did: Did) -> None:
    windows = _windows(events, victim)
    assert len(windows) >= K
    assert all(any(did(e, victim) for e in w) for w in windows)
    assert not _starvation(events, victim), _starvation(events, victim)[0].detail
    assert victim.state is not JobState.FAULTED


def _assert_starves(victim: Job, events: Sequence[Event], did: Did) -> None:
    windows = _windows(events, victim)[:K]
    assert len(windows) == K
    assert all(any(did(e, victim) for e in w) for w in windows)
    assert not any(_progress(e, victim) for e in events)
    assert _preemptions_before_starvation(events, victim) == K + 1


def _wrote(pipe: PipeName, *, latched: bool = False) -> Did:
    return lambda e, v: (
        isinstance(e, PipeWritten) and e.job == v.job_id and e.pipe == pipe and e.latched is latched
    )


def _faulted(kind: FaultKind) -> Did:
    return lambda e, v: isinstance(e, FaultRaised) and e.job == v.job_id and e.fault is kind


def _answered(*, allowed: bool) -> Did:
    return lambda e, v: isinstance(e, GateAnswered) and e.job == v.job_id and e.allowed is allowed


def _decoded(e: Event, v: Job) -> bool:
    return isinstance(e, Decoded) and e.job == v.job_id


def _preemptions(events: Sequence[Event], victim: Job) -> int:
    return sum(1 for e in events if isinstance(e, JobPreempted) and e.job == victim.job_id)


# --- progress ------------------------------------------------------------------


def test_a_write_to_an_ordinary_pipe_is_progress() -> None:
    did = _wrote(OUT)
    victim, events = _run(_script([_write("stdout")] * STEPS), stepped=did)
    _assert_never_starves(victim, events, did)


def test_a_write_to_a_sink_is_progress() -> None:
    did = _wrote(SINK)
    victim, events = _run(_script([_write("sink")] * STEPS), stepped=did)
    _assert_never_starves(victim, events, did)


def test_a_latched_actuator_write_is_progress() -> None:
    did = _wrote(DOOR, latched=True)
    victim, events = _run(_script([_write("tools")] * STEPS), stepped=did)
    _assert_never_starves(victim, events, did)


def test_a_write_sent_over_a_link_is_progress() -> None:
    def did(e: Event, v: Job) -> bool:
        return isinstance(e, FrameSent) and e.pipe == UPLINK and e.tokens > 0

    victim, events = _run(_script([_write("uplink")] * STEPS), stepped=did, link=True)
    _assert_never_starves(victim, events, did)
    assert not [e for e in events if isinstance(e, PipeWritten) and e.pipe == UPLINK]


def test_a_gated_write_the_guard_allows_is_progress() -> None:
    did = _wrote(DOOR, latched=True)
    victim, events = _run(_script([GATED, WORK] * STEPS), stepped=did, guard="allow")
    _assert_never_starves(victim, events, did)
    assert all(any(_answered(allowed=True)(e, victim) for e in w) for w in _windows(events, victim))


def test_a_read_that_gets_data_is_progress() -> None:
    def did(e: Event, v: Job) -> bool:
        return isinstance(e, PipeReadEvent) and e.job == v.job_id and e.tokens > 0

    victim, events = _run(_script([{"read": "stdin"}] * STEPS), stepped=did, feed=True)
    _assert_never_starves(victim, events, did)
    assert not [e for e in events if isinstance(e, JobBlocked) and e.job == victim.job_id]


def test_acquiring_a_lock_is_progress() -> None:
    def did(e: Event, v: Job) -> bool:
        return isinstance(e, ResourceAcquired) and e.job == v.job_id

    steps = [{"acquire": f"lock.{i}"} for i in range(LOCKS)]
    victim, events = _run(_script(steps), stepped=did)
    _assert_never_starves(victim, events, did)
    acquired = [
        e.resource for e in events if isinstance(e, ResourceAcquired) and e.job == victim.job_id
    ]
    assert len(acquired) == len(set(acquired)), "each acquisition takes a lock not yet held"


def test_releasing_a_held_lock_is_progress() -> None:
    def did(e: Event, v: Job) -> bool:
        return isinstance(e, ResourceReleased) and e.job == v.job_id

    steps = [{"acquire": f"lock.{i}"} for i in range(LOCKS)]
    steps += [{"release": f"lock.{i}"} for i in range(LOCKS)]
    victim, events = _run(_script(steps), stepped=did)
    _assert_never_starves(victim, events, did)
    first = next(i for i, e in enumerate(events) if isinstance(e, JobPreempted))
    assert not [e for e in events[first:] if isinstance(e, ResourceAcquired)]


def test_spawning_a_child_is_progress() -> None:
    def did(e: Event, v: Job) -> bool:
        return isinstance(e, JobSpawned) and e.parent == v.job_id

    victim, events = _run(_script([{"spawn": "child"}] * STEPS), stepped=did)
    _assert_never_starves(victim, events, did)


# --- not progress --------------------------------------------------------------


def test_tokens_alone_are_not_progress() -> None:
    victim, events = _run(_script([WORK] * STEPS), stepped=_decoded)
    _assert_starves(victim, events, _decoded)
    assert victim.state is JobState.FAULTED


def test_a_refused_write_is_not_progress() -> None:
    did = _faulted(FaultKind.CAPABILITY)
    victim, events = _run(_script([_write("elsewhere")] * STEPS), stepped=did, on_fault="retry")
    _assert_starves(victim, events, did)


def test_a_malformed_command_is_not_progress() -> None:
    malformed = Step(request=MachineRequest(op=OpKind.MALFORMED, text="frobnicate"))
    exit_step = Step(request=MachineRequest(op=OpKind.EXIT))
    did = _faulted(FaultKind.MALFORMED)
    victim, events = _run(
        Script(steps=(malformed,) * STEPS + (exit_step,)), stepped=did, on_fault="retry"
    )
    _assert_starves(victim, events, did)


def test_a_gated_write_the_guard_vetoes_is_not_progress() -> None:
    did = _answered(allowed=False)
    victim, events = _run(
        _script([GATED, WORK] * STEPS), stepped=did, guard="veto", on_fault="retry"
    )
    _assert_starves(victim, events, did)
    assert all(
        any(_faulted(FaultKind.GATE)(e, victim) for e in w) for w in _windows(events, victim)
    )


def test_losing_a_body_is_not_progress() -> None:
    revoked: list[Disembodied] = []

    def between(kernel: Kernel, victim: Job, events: list[Event]) -> None:
        if not revoked and _preemptions(events, victim) == K // 2:
            kernel.revoke_lease(CARRIER.name)
            revoked.extend(e for e in events if isinstance(e, Disembodied))

    victim, events = _run(_script([WORK] * STEPS), stepped=_decoded, body=True, between=between)
    assert [e.job for e in revoked] == [victim.job_id]
    assert _preemptions_before_starvation(events, victim) == K + 1


def test_getting_a_body_back_is_not_progress() -> None:
    """A more urgent job takes the victim's body and waits on a pipe; the victim queues
    for the body and is granted it when that job finishes."""
    done: set[str] = set()

    def between(kernel: Kernel, victim: Job, events: list[Event]) -> None:
        if "spawned" not in done and _preemptions(events, victim) == K // 2:
            kernel.spawn(DescriptorName("rival"))
            done.add("spawned")
        queued = any(isinstance(e, ResourceBlocked) and e.job == victim.job_id for e in events)
        if queued and "woken" not in done:
            kernel.deliver(RIVAL_IN, "go")
            done.add("woken")

    victim, events = _run(_script([WORK] * STEPS), stepped=_decoded, body=True, between=between)
    regained = [
        e
        for e in events
        if isinstance(e, ResourceAcquired)
        and e.job == victim.job_id
        and e.resource_kind is ResourceKind.LEASE
    ]
    assert len(regained) == 1
    assert _preemptions_before_starvation(events, victim) == K + 1
