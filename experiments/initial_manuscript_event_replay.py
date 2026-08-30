"""Continuous two-GPU event replay for the initial-manuscript HBTASP track."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np

from experiments.mandatory_terminal_registry import MandatoryTerminalRegistry
from experiments.initial_manuscript_hbtasp import (
    GPU_WCET, TAMB, THERMAL_B, VOLTAGES, RegionJob, StageOneAssignment,
    assign_mandatory, assign_optional_manuscript, end_temperature,
    select_stage_two, steady_temperature,
)


@dataclass
class ProcessorState:
    queue: list[StageOneAssignment]
    busy: object | None = None
    busy_until: float = 0.0
    temperature: float = TAMB
    temperature_time: float = 0.0


@dataclass
class RunningExecution:
    decision: object
    finish: float
    actual_duration: float
    actual_end_temperature: float
    overrun_factor: float
    iit_celsius_seconds: float


def plant_steady_temperature(voltage: float, response_factor: float = 1.0,
                             ambient_c: float = TAMB) -> float:
    """Plant steady state under a structured effective-response mismatch.

    The nominal manuscript model is recovered exactly at resistance_factor=1
    and ambient_c=TAMB.  The scheduler always retains the nominal model; these
    parameters affect only the replayed plant. The response factor is not
    interpreted as a uniquely identified physical R_th because the manuscript
    publishes the absorbed alpha and B parameters.
    """
    if response_factor <= 0:
        raise ValueError("response_factor must be positive")
    return ambient_c + response_factor * (steady_temperature(voltage) - TAMB)


def plant_end_temperature(start: float, voltage: float, duration: float,
                          response_factor: float = 1.0,
                          ambient_c: float = TAMB) -> float:
    tss = plant_steady_temperature(voltage, response_factor, ambient_c)
    beta = THERMAL_B / response_factor
    return tss + (start - tss) * np.exp(-beta * duration)


def temperature_excursion_integral(start: float, voltage: float, duration: float,
                                   threshold: float = 60.0,
                                   response_factor: float = 1.0,
                                   ambient_c: float = TAMB) -> float:
    """Exact integral of max(T(t)-threshold, 0) for one RC segment."""
    if duration <= 0:
        return 0.0
    beta = THERMAL_B / response_factor
    tss = plant_steady_temperature(voltage, response_factor, ambient_c)
    end = tss + (start - tss) * np.exp(-beta * duration)

    def excess_integral(a: float, b: float) -> float:
        # Integral_a^b [T(t)-threshold] dt for the segment anchored at t=0.
        return ((tss - threshold) * (b - a)
                + (start - tss) * (np.exp(-beta*a) - np.exp(-beta*b)) / beta)

    if start <= threshold and end <= threshold:
        return 0.0
    if start >= threshold and end >= threshold:
        return float(max(0.0, excess_integral(0.0, duration)))
    ratio = (threshold - tss) / (start - tss)
    crossing = -np.log(ratio) / beta
    crossing = float(min(duration, max(0.0, crossing)))
    if start < threshold < end:
        return float(max(0.0, excess_integral(crossing, duration)))
    return float(max(0.0, excess_integral(0.0, crossing)))


def build_periodic_releases(
    pool: Sequence[dict], period_ms: int, lines: int, epochs: int, seed: int,
    all_regions_mandatory: bool = False,
    mandatory_count: int = 1,
) -> dict[float, list[RegionJob]]:
    if mandatory_count not in (1, 2, 3, 4):
        raise ValueError("mandatory_count must be in {1,2,3,4}")
    rng = Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    period = period_ms / 1000.0
    releases: dict[float, list[RegionJob]] = {}
    cursor = 0
    for epoch in range(epochs):
        release = epoch * period
        jobs = []
        for line in range(lines):
            item = pool[order[cursor % len(order)]]
            cursor += 1
            image_uid = f"e{epoch}:l{line}:{item['image_id']}"
            ranked = sorted(range(4), key=lambda r: (-float(item["weights"][r]), r))
            mandatory_regions = set(ranked[:4 if all_regions_mandatory else mandatory_count])
            for region in range(4):
                jobs.append(RegionJob(
                    f"{image_uid}:r{region}", image_uid, release,
                    release + period, float(item["weights"][region]),
                    region in mandatory_regions,
                    str(item["image_id"]), region,
                ))
        releases[release] = jobs
    return releases


def build_poisson_releases(
    pool: Sequence[dict], period_ms: int, lines: int, epochs: int, seed: int,
    all_regions_mandatory: bool = False,
    mandatory_count: int = 1,
) -> dict[float, list[RegionJob]]:
    """Independent Poisson streams with mean inter-arrival equal to period_ms."""
    if mandatory_count not in (1, 2, 3, 4):
        raise ValueError("mandatory_count must be in {1,2,3,4}")
    rng = Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    period = period_ms / 1000.0
    horizon = epochs * period
    releases: dict[float, list[RegionJob]] = {}
    cursor = 0
    for line in range(lines):
        release = 0.0
        arrival_index = 0
        while True:
            release += rng.expovariate(1.0 / period)
            if release >= horizon:
                break
            item = pool[order[cursor % len(order)]]
            cursor += 1
            image_uid = f"p{arrival_index}:l{line}:{item['image_id']}"
            jobs = []
            ranked = sorted(range(4), key=lambda r: (-float(item["weights"][r]), r))
            mandatory_regions = set(ranked[:4 if all_regions_mandatory else mandatory_count])
            for region in range(4):
                jobs.append(RegionJob(
                    f"{image_uid}:r{region}", image_uid, release,
                    release + period, float(item["weights"][region]),
                    region in mandatory_regions,
                    str(item["image_id"]), region,
                ))
            releases.setdefault(release, []).extend(jobs)
            arrival_index += 1
    return releases


def run_continuous_hbtasp(
    pool: Sequence[dict], period_ms: int, lines: int, epochs: int, seed: int,
    budget_mode: str = "manuscript_l5",
    network_mode: str = "dynamic",
    overrun_probability: float = 0.0,
    overrun_min_extra: float = 0.0,
    overrun_max_extra: float = 0.0,
    overrun_seed: int | None = None,
    forced_overrun_times: Sequence[float] = (),
    forced_overrun_extra: float = 0.0,
    forced_overrun_min_voltage: float = 0.0,
    release_mode: str = "periodic",
    priority_overhead_ms: float = 0.0,
    all_regions_mandatory: bool = False,
    enable_productive_cooling: bool = True,
    fixed_voltage_index: int | None = None,
    enforce_thermal_feasibility: bool = True,
    plant_thermal_response_factor: float = 1.0,
    plant_ambient_c: float = TAMB,
    mandatory_count: int = 1,
) -> tuple[dict, list[dict]]:
    if plant_thermal_response_factor <= 0:
        raise ValueError("plant_thermal_response_factor must be positive")
    if release_mode == "periodic":
        releases = build_periodic_releases(
            pool, period_ms, lines, epochs, seed, all_regions_mandatory,
            mandatory_count,
        )
    elif release_mode == "poisson":
        releases = build_poisson_releases(
            pool, period_ms, lines, epochs, seed, all_regions_mandatory,
            mandatory_count,
        )
    else:
        raise ValueError(f"unsupported release_mode: {release_mode}")
    if priority_overhead_ms < 0:
        raise ValueError("priority_overhead_ms must be non-negative")
    if priority_overhead_ms:
        delay = priority_overhead_ms / 1000.0
        delayed: dict[float, list[RegionJob]] = {}
        for _original_release, jobs in releases.items():
            for job in jobs:
                shifted = RegionJob(
                    job.uid, job.image_uid, job.release + delay, job.deadline,
                    job.weight, job.mandatory, job.source_image_id,
                    job.region_index,
                )
                delayed.setdefault(shifted.release, []).append(shifted)
        releases = delayed
    release_times = sorted(releases)
    release_index = 0
    processors = [
        ProcessorState([], temperature=plant_ambient_c),
        ProcessorState([], temperature=plant_ambient_c),
    ]
    overrun_rng = Random(seed if overrun_seed is None else overrun_seed)
    pending_forced_overruns = list(sorted(forced_overrun_times))
    trace: list[dict] = []
    terminal_registry = MandatoryTerminalRegistry.from_release_batches(releases.values())
    counters = {
        "total_regions": sum(len(jobs) for jobs in releases.values()),
        "mandatory_infeasible": 0,
        "optional_skipped": 0,
        "dispatch_infeasible": 0,
        "expired_waiting": 0,
        "completed": 0,
        "deadline_misses": 0,
        "thermal_violations": 0,
    }

    def available_times(now: float) -> list[float]:
        return [state.busy_until if state.busy is not None else now
                for state in processors]

    def cool_to(state: ProcessorState, now: float) -> None:
        if state.busy is None and now > state.temperature_time:
            state.temperature = plant_end_temperature(
                state.temperature, VOLTAGES[0], now - state.temperature_time,
                plant_thermal_response_factor, plant_ambient_c,
            )
            state.temperature_time = now

    while release_index < len(release_times) or any(
        state.busy is not None or state.queue for state in processors
    ):
        next_release = (release_times[release_index]
                        if release_index < len(release_times) else float("inf"))
        next_completion = min(
            (state.busy_until for state in processors if state.busy is not None),
            default=float("inf"),
        )
        now = min(next_release, next_completion)
        if now == float("inf"):
            break

        # Completion is processed before a simultaneous release.
        for processor, state in enumerate(processors):
            if state.busy is not None and abs(state.busy_until - now) <= 1e-12:
                running = state.busy
                decision = running.decision
                state.temperature = running.actual_end_temperature
                state.temperature_time = now
                state.busy = None
                counters["completed"] += 1
                missed = now > decision.assignment.job.deadline + 1e-12
                counters["deadline_misses"] += int(missed)
                counters["thermal_violations"] += int(
                    state.temperature > 60.0 + 1e-10
                )
                terminal_state = None
                if decision.assignment.job.mandatory:
                    terminal_state = terminal_registry.transition(
                        decision.assignment.job.uid,
                        "LATE_COMPLETE" if missed else "ON_TIME",
                    )
                trace.append({
                    "event": "complete", "time": now, "processor": processor,
                    "uid": decision.assignment.job.uid,
                    "mandatory": decision.assignment.job.mandatory,
                    "level": decision.level + 1,
                    "voltage": VOLTAGES[decision.voltage_index],
                    "temperature": state.temperature, "deadline_miss": missed,
                    "image_uid": decision.assignment.job.image_uid,
                    "source_image_id": decision.assignment.job.source_image_id,
                    "region_index": decision.assignment.job.region_index,
                    "weight": decision.assignment.job.weight,
                    "terminal_state": terminal_state,
                    "planned_duration": decision.duration,
                    "actual_duration": running.actual_duration,
                    "overrun_factor": running.overrun_factor,
                    "iit_celsius_seconds": running.iit_celsius_seconds,
                })

        if release_index < len(release_times) and abs(next_release - now) <= 1e-12:
            new_jobs = releases[next_release]
            queues = [state.queue for state in processors]
            queues, infeasible = assign_mandatory(
                new_jobs, queues, available_times(now),
                mandatory_level=2 if network_mode == "fixed_l3" else 4,
                budget_level=2 if network_mode == "fixed_l3" else 4,
            )
            counters["mandatory_infeasible"] += len(infeasible)
            queues, audit = assign_optional_manuscript(
                new_jobs, queues, available_times(now),
                allowed_levels=(2,) if network_mode == "fixed_l3" else (0,1,2,3,4),
                budget_level=2 if network_mode == "fixed_l3" else 4,
                enforce_initial_mandatory_l5=(
                    network_mode != "fixed_l3"
                    and budget_mode != "assigned_level_sensitivity"
                ),
                budget_from_selected_level=(
                    budget_mode == "assigned_level_sensitivity"
                ),
            )
            counters["optional_skipped"] += len(audit.skipped)
            for job in infeasible:
                terminal_state = terminal_registry.transition(job.uid, "PRE_REJECT")
                trace.append({
                    "event": "mandatory_infeasible", "time": now,
                    "uid": job.uid, "image_uid": job.image_uid,
                    "source_image_id": job.source_image_id,
                    "region_index": job.region_index, "mandatory": True,
                    "weight": job.weight,
                    "terminal_state": terminal_state,
                })
            new_by_uid = {job.uid: job for job in new_jobs}
            for uid in audit.skipped:
                job = new_by_uid[uid]
                trace.append({
                    "event": "optional_skipped", "time": now,
                    "uid": job.uid, "image_uid": job.image_uid,
                    "source_image_id": job.source_image_id,
                    "region_index": job.region_index, "mandatory": False,
                    "weight": job.weight,
                })
            for processor, queue in enumerate(queues):
                processors[processor].queue = queue
            trace.append({
                "event": "release", "time": now,
                "mandatory_infeasible": len(infeasible),
                "optional_skipped": len(audit.skipped),
            })
            release_index += 1

        # Dispatch on every idle processor after all state changes at this time.
        progress = True
        while progress:
            progress = False
            for processor, state in enumerate(processors):
                if state.busy is not None or not state.queue:
                    continue
                cool_to(state, now)
                head = state.queue[0]
                if now > head.job.deadline + 1e-12:
                    state.queue.pop(0)
                    counters["expired_waiting"] += 1
                    counters["deadline_misses"] += 1
                    terminal_state = None
                    if head.job.mandatory:
                        terminal_state = terminal_registry.transition(
                            head.job.uid, "WAIT_EXPIRE"
                        )
                    trace.append({
                        "event": "expired", "time": now,
                        "processor": processor, "uid": head.job.uid,
                        "mandatory": head.job.mandatory,
                        "image_uid": head.job.image_uid,
                        "source_image_id": head.job.source_image_id,
                        "region_index": head.job.region_index,
                        "weight": head.job.weight,
                        "terminal_state": terminal_state,
                    })
                    progress = True
                    continue
                decision = select_stage_two(
                    head, state.queue, state.temperature,
                    fixed_level=2 if network_mode == "fixed_l3" else None,
                    enable_productive_cooling=enable_productive_cooling,
                    fixed_voltage_index=fixed_voltage_index,
                    enforce_thermal_feasibility=enforce_thermal_feasibility,
                )
                state.queue.pop(0)
                if decision is None:
                    counters["dispatch_infeasible"] += 1
                    terminal_state = None
                    if head.job.mandatory:
                        terminal_state = terminal_registry.transition(
                            head.job.uid, "PRE_REJECT"
                        )
                    trace.append({
                        "event": "dispatch_infeasible", "time": now,
                        "processor": processor, "uid": head.job.uid,
                        "mandatory": head.job.mandatory,
                        "image_uid": head.job.image_uid,
                        "source_image_id": head.job.source_image_id,
                        "region_index": head.job.region_index,
                        "weight": head.job.weight,
                        "terminal_state": terminal_state,
                    })
                    progress = True
                    continue
                factor = 1.0
                forced = bool(
                    pending_forced_overruns
                    and now + 1e-12 >= pending_forced_overruns[0]
                    and VOLTAGES[decision.voltage_index] >= forced_overrun_min_voltage
                )
                if forced:
                    pending_forced_overruns.pop(0)
                    factor += forced_overrun_extra
                elif overrun_probability > 0 and overrun_rng.random() < overrun_probability:
                    factor += overrun_rng.uniform(
                        overrun_min_extra, overrun_max_extra
                    )
                actual_duration = decision.duration * factor
                actual_end = plant_end_temperature(
                    state.temperature, VOLTAGES[decision.voltage_index], actual_duration,
                    plant_thermal_response_factor, plant_ambient_c,
                )
                iit = temperature_excursion_integral(
                    state.temperature, VOLTAGES[decision.voltage_index], actual_duration,
                    response_factor=plant_thermal_response_factor,
                    ambient_c=plant_ambient_c,
                )
                state.busy = RunningExecution(
                    decision, now + actual_duration, actual_duration,
                    actual_end, factor, iit,
                )
                state.busy_until = now + actual_duration
                trace.append({
                    "event": "dispatch", "time": now,
                    "processor": processor, "uid": head.job.uid,
                    "mandatory": head.job.mandatory,
                    "level": decision.level + 1,
                    "voltage": VOLTAGES[decision.voltage_index],
                    "temperature": state.temperature,
                    "planned_duration": decision.duration,
                    "overrun_factor": factor,
                })

    counters.update(terminal_registry.finalize())
    counters.update({
        "period_ms": period_ms, "lines": lines, "epochs": epochs,
        "seed": seed, "budget_mode": budget_mode,
        "network_mode": network_mode,
        "release_mode": release_mode,
        "priority_overhead_ms": priority_overhead_ms,
        "all_regions_mandatory": all_regions_mandatory,
        "mandatory_count": 4 if all_regions_mandatory else mandatory_count,
        "enable_productive_cooling": enable_productive_cooling,
        "fixed_voltage_index": fixed_voltage_index,
        "enforce_thermal_feasibility": enforce_thermal_feasibility,
        "plant_thermal_response_factor": plant_thermal_response_factor,
        "plant_ambient_c": plant_ambient_c,
        "overrun_probability": overrun_probability,
        "overrun_min_extra": overrun_min_extra,
        "overrun_max_extra": overrun_max_extra,
        "forced_overrun_count": len(forced_overrun_times),
    })
    return counters, trace
