"""Unified-release adapters for the submitted Static-V, ESATD, and HEAT baselines.

The adapters preserve the decision rules of the available baseline implementations
while sharing the HBTASP release pool, job identifiers, terminal registry, T4
profiles, and trace schema. Early-exit skipping is deliberately disabled by the
frozen Overall V11 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from experiments.initial_manuscript_event_replay import (
    build_periodic_releases, run_continuous_hbtasp,
)
from experiments.initial_manuscript_hbtasp import (
    GPU_WCET, TAMB, VOLTAGES, end_temperature,
)
from experiments.mandatory_terminal_registry import MandatoryTerminalRegistry


L3 = 2
REFERENCE_VOLTAGE_INDEX = 2
T_HOT = 55.0


@dataclass
class CoreState:
    free: float = 0.0
    temperature: float = TAMB
    temperature_time: float = 0.0


def _duration(core: int, voltage_index: int, level_index: int = L3) -> float:
    return GPU_WCET[core][level_index] * VOLTAGES[REFERENCE_VOLTAGE_INDEX] / VOLTAGES[voltage_index]


def _cool_to(state: CoreState, when: float) -> None:
    if when > state.temperature_time:
        state.temperature = end_temperature(
            state.temperature, VOLTAGES[0], when - state.temperature_time
        )
        state.temperature_time = when


def _terminal_event(registry, job, finish: float | None, event: str, **extra):
    terminal_state = None
    # Deadline credit applies to every returned region. The mandatory registry
    # is a separate service audit; restricting this flag to mandatory jobs would
    # incorrectly credit late optional predictions during complete-image replay.
    deadline_miss = (
        event == "complete" and finish is not None
        and finish > job.deadline + 1e-12
    )
    if job.mandatory:
        if event == "complete":
            terminal_state = registry.transition(
                job.uid, "LATE_COMPLETE" if deadline_miss else "ON_TIME"
            )
        elif event == "expired":
            terminal_state = registry.transition(job.uid, "WAIT_EXPIRE")
        else:
            terminal_state = registry.transition(job.uid, "PRE_REJECT")
    row = {
        "event": event, "uid": job.uid, "mandatory": job.mandatory,
        "image_uid": job.image_uid, "source_image_id": job.source_image_id,
        "region_index": job.region_index, "weight": job.weight,
        "terminal_state": terminal_state,
    }
    if event == "complete":
        row["deadline_miss"] = deadline_miss
    row.update(extra)
    return row


def run_static_v(pool: Sequence[dict], period_ms: int, lines: int,
                 epochs: int, seed: int):
    """Formal HBTASP assignment/dynamic-level policy at fixed 0.80 V."""
    return run_continuous_hbtasp(
        pool, period_ms, lines, epochs, seed,
        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
        enable_productive_cooling=False, fixed_voltage_index=REFERENCE_VOLTAGE_INDEX,
        enforce_thermal_feasibility=False,
    )


def run_esatd_l3(pool: Sequence[dict], period_ms: int, lines: int,
                  epochs: int, seed: int):
    """Port of the corrected ESATD fixed-L3 lowest-feasible-voltage policy."""
    return run_esatd_fixed(pool, period_ms, lines, epochs, seed, 3)


def run_esatd_fixed(pool: Sequence[dict], period_ms: int, lines: int,
                    epochs: int, seed: int, level: int):
    """ESATD fixed-path lowest-feasible-voltage policy for L1--L5."""
    level_index = level - 1
    releases = build_periodic_releases(pool, period_ms, lines, epochs, seed)
    registry = MandatoryTerminalRegistry.from_release_batches(releases.values())
    cores = [CoreState(), CoreState()]
    trace: list[dict] = []
    counters = {"total_regions": lines * epochs * 4, "completed": 0,
                "expired_waiting": 0, "admission_infeasible": 0,
                "deadline_misses": 0, "thermal_violations": 0}

    for release, jobs in sorted(releases.items()):
        # The corrected sensitivity implementation gives mandatory work stable
        # priority when implicit deadlines tie.
        for job in sorted(jobs, key=lambda x: (x.deadline, not x.mandatory, x.uid)):
            best = None
            for core, state in enumerate(cores):
                start = max(release, state.free)
                remaining = job.deadline - start
                if remaining <= 1e-12:
                    continue
                voltage_index = next(
                    (v for v in range(len(VOLTAGES))
                     if _duration(core, v, level_index) <= remaining + 1e-12),
                    len(VOLTAGES) - 1,
                )
                candidate = (state.free, core, voltage_index, start,
                             _duration(core, voltage_index, level_index))
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                counters["admission_infeasible"] += 1
                trace.append(_terminal_event(
                    registry, job, None, "dispatch_infeasible", time=release
                ))
                continue
            _load, core, voltage_index, start, duration = best
            state = cores[core]
            _cool_to(state, start)
            finish = start + duration
            start_temp = state.temperature
            state.temperature = end_temperature(
                state.temperature, VOLTAGES[voltage_index], duration
            )
            state.temperature_time = finish
            state.free = finish
            counters["completed"] += 1
            missed = finish > job.deadline + 1e-12
            counters["deadline_misses"] += int(missed)
            counters["thermal_violations"] += int(state.temperature > 60.0 + 1e-10)
            trace.append({"event": "dispatch", "time": start, "processor": core,
                          "uid": job.uid, "mandatory": job.mandatory, "level": level,
                          "voltage": VOLTAGES[voltage_index], "temperature": start_temp})
            trace.append(_terminal_event(
                registry, job, finish, "complete", time=finish, processor=core,
                level=level, voltage=VOLTAGES[voltage_index], temperature=state.temperature,
            ))

    counters.update(registry.finalize())
    counters.update({"period_ms": period_ms, "lines": lines, "epochs": epochs,
                     "seed": seed, "scheduler": f"ESATD-L{level}",
                     "release_mode": "periodic", "early_exit": False})
    return counters, trace


def _heat_voltage(task_count: int, core: int, period: float,
                  current_temperature: float,
                  temperature_compensation: str = "source_raise") -> int:
    total_fixed = task_count * GPU_WCET[core][L3]
    required = max(0.3, min(1.0, total_fixed / period))
    target = VOLTAGES[0] + required * (VOLTAGES[-1] - VOLTAGES[0])
    voltage_index = min(range(len(VOLTAGES)), key=lambda v: abs(VOLTAGES[v] - target))
    if temperature_compensation == "source_raise":
        if current_temperature > T_HOT:
            voltage_index = min(len(VOLTAGES) - 1, voltage_index + 1)
        elif current_temperature < TAMB + 5:
            voltage_index = max(0, voltage_index - 1)
    elif temperature_compensation == "safety_lower":
        if current_temperature > T_HOT:
            voltage_index = max(0, voltage_index - 1)
        elif current_temperature < TAMB + 5:
            voltage_index = min(len(VOLTAGES) - 1, voltage_index + 1)
    elif temperature_compensation != "none":
        raise ValueError(
            "temperature_compensation must be source_raise, none, or safety_lower"
        )
    return voltage_index


def run_heat_l3(pool: Sequence[dict], period_ms: int, lines: int,
                 epochs: int, seed: int,
                 temperature_compensation: str = "source_raise"):
    """Corrected port of the intended HEAT fixed-L3 two-core pipeline."""
    releases = build_periodic_releases(pool, period_ms, lines, epochs, seed)
    registry = MandatoryTerminalRegistry.from_release_batches(releases.values())
    cores = [CoreState(), CoreState()]
    trace: list[dict] = []
    counters = {"total_regions": lines * epochs * 4, "completed": 0,
                "expired_waiting": 0, "admission_infeasible": 0,
                "deadline_misses": 0, "thermal_violations": 0}
    period = period_ms / 1000.0

    for release, jobs in sorted(releases.items()):
        # SCHEDULE-DESIGN for the two-core fixed-path case: fill the preferred
        # (faster) core, then the second core, and finally place any overload on
        # the core with more residual capacity. This corrects the available
        # implementation's loop-index defect that could silently lose tasks.
        spare = [period, period]
        assigned = {0: [], 1: []}
        cursor = 0
        while cursor < len(jobs):
            job = jobs[cursor]
            if GPU_WCET[0][L3] <= spare[0] + 1e-12:
                assigned[0].append(job)
                spare[0] -= GPU_WCET[0][L3]
                cursor += 1
            else:
                break
        tail = len(jobs) - 1
        while tail >= cursor:
            job = jobs[tail]
            if GPU_WCET[1][L3] <= spare[1] + 1e-12:
                assigned[1].append(job)
                spare[1] -= GPU_WCET[1][L3]
                tail -= 1
            else:
                break
        for job in jobs[cursor:tail + 1]:
            core = 0 if spare[0] >= spare[1] else 1
            assigned[core].append(job)
            spare[core] -= GPU_WCET[core][L3]

        for core, jobs_on_core in assigned.items():
            if not jobs_on_core:
                continue
            state = cores[core]
            voltage_index = _heat_voltage(
                len(jobs_on_core), core, period, state.temperature,
                temperature_compensation,
            )
            for job in jobs_on_core:
                start = max(release, state.free)
                _cool_to(state, start)
                duration = _duration(core, voltage_index)
                finish = start + duration
                start_temp = state.temperature
                state.temperature = end_temperature(
                    state.temperature, VOLTAGES[voltage_index], duration
                )
                state.temperature_time = finish
                state.free = finish
                counters["completed"] += 1
                missed = finish > job.deadline + 1e-12
                counters["deadline_misses"] += int(missed)
                counters["thermal_violations"] += int(
                    state.temperature > 60.0 + 1e-10
                )
                trace.append({"event": "dispatch", "time": start,
                              "processor": core, "uid": job.uid,
                              "mandatory": job.mandatory, "level": 3,
                              "voltage": VOLTAGES[voltage_index],
                              "temperature": start_temp})
                trace.append(_terminal_event(
                    registry, job, finish, "complete", time=finish,
                    processor=core, level=3, voltage=VOLTAGES[voltage_index],
                    temperature=state.temperature,
                ))

    counters.update(registry.finalize())
    counters.update({"period_ms": period_ms, "lines": lines, "epochs": epochs,
                     "seed": seed, "scheduler": "HEAT-L3",
                     "release_mode": "periodic", "early_exit": False,
                     "temperature_compensation": temperature_compensation})
    return counters, trace
