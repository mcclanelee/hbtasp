"""Independent global-EDF baselines for the initial-manuscript evidence track."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

from experiments.initial_manuscript_event_replay import build_periodic_releases
from experiments.mandatory_terminal_registry import MandatoryTerminalRegistry
from experiments.initial_manuscript_hbtasp import (
    GPU_WCET, TAMB, RegionJob, end_temperature,
)


BASELINE_VOLTAGE = 0.8


@dataclass
class Running:
    job: RegionJob
    level: int
    finish: float


@dataclass
class EdfProcessor:
    running: Running | None = None
    temperature: float = TAMB
    temperature_time: float = 0.0


def _reservation_feasible(job, processor, level, now, ready, processors) -> bool:
    available = [p.running.finish if p.running else now for p in processors]
    available[processor] = now + GPU_WCET[processor][level]
    if available[processor] > job.deadline + 1e-12:
        return False
    for _deadline, _order, queued in sorted(ready):
        queued_level = 4 if queued.mandatory else 0
        choices = [(max(available[p], queued.release) + GPU_WCET[p][queued_level], p)
                   for p in range(2)]
        finish, chosen = min(choices)
        if finish > queued.deadline + 1e-12:
            return False
        available[chosen] = finish
    return True


def _select_level(job: RegionJob, processor: int, now: float, mode: str,
                  ready=(), processors=()) -> int | None:
    if mode == "fixed_l3":
        candidates = (2,)
    elif mode in ("dynamic", "dynamic_reservation"):
        candidates = (4,) if job.mandatory else range(4, -1, -1)
    else:
        raise ValueError(f"unknown EDF mode: {mode}")
    for level in candidates:
        basic = now + GPU_WCET[processor][level] <= job.deadline + 1e-12
        reserved = (mode != "dynamic_reservation" or
                    _reservation_feasible(job, processor, level, now, ready, processors))
        if basic and reserved:
            return level
    return None


def run_edf(
    pool: Sequence[dict], period_ms: int, lines: int, epochs: int, seed: int,
    mode: str,
) -> tuple[dict, list[dict]]:
    releases = build_periodic_releases(pool, period_ms, lines, epochs, seed)
    release_times = sorted(releases)
    release_index = 0
    counter = 0
    ready = []
    processors = [EdfProcessor(), EdfProcessor()]
    trace = []
    terminal_registry = MandatoryTerminalRegistry.from_release_batches(releases.values())
    result = {"total_regions": lines * epochs * 4, "completed": 0,
              "expired_waiting": 0, "admission_infeasible": 0,
              "deadline_misses": 0, "thermal_violations": 0}

    while release_index < len(release_times) or ready or any(p.running for p in processors):
        next_release = (release_times[release_index]
                        if release_index < len(release_times) else float("inf"))
        next_completion = min((p.running.finish for p in processors if p.running),
                              default=float("inf"))
        now = min(next_release, next_completion)
        if now == float("inf"):
            break

        for processor, state in enumerate(processors):
            if state.running and abs(state.running.finish - now) <= 1e-12:
                running = state.running
                state.temperature = end_temperature(
                    state.temperature, BASELINE_VOLTAGE,
                    GPU_WCET[processor][running.level],
                )
                state.temperature_time = now
                state.running = None
                missed = now > running.job.deadline + 1e-12
                result["completed"] += 1
                result["deadline_misses"] += int(missed)
                result["thermal_violations"] += int(state.temperature > 60.0 + 1e-10)
                terminal_state = None
                if running.job.mandatory:
                    terminal_state = terminal_registry.transition(
                        running.job.uid, "LATE_COMPLETE" if missed else "ON_TIME"
                    )
                trace.append({"event": "complete", "time": now, "processor": processor,
                    "uid": running.job.uid, "image_uid": running.job.image_uid,
                    "source_image_id": running.job.source_image_id,
                    "region_index": running.job.region_index,
                    "mandatory": running.job.mandatory, "level": running.level + 1,
                    "voltage": BASELINE_VOLTAGE, "temperature": state.temperature,
                    "deadline_miss": missed, "terminal_state": terminal_state})
                trace[-1]["weight"] = running.job.weight

        if release_index < len(release_times) and abs(next_release - now) <= 1e-12:
            for job in releases[next_release]:
                heapq.heappush(ready, (job.deadline, counter, job))
                counter += 1
            release_index += 1

        for processor, state in enumerate(processors):
            while state.running is None and ready:
                _deadline, _order, job = heapq.heappop(ready)
                if now > job.deadline + 1e-12:
                    result["expired_waiting"] += 1
                    result["deadline_misses"] += 1
                    terminal_state = None
                    if job.mandatory:
                        terminal_state = terminal_registry.transition(job.uid, "WAIT_EXPIRE")
                    trace.append({"event": "expired", "time": now,
                        "processor": processor, "uid": job.uid,
                        "image_uid": job.image_uid,
                        "source_image_id": job.source_image_id,
                        "region_index": job.region_index,
                        "mandatory": job.mandatory, "terminal_state": terminal_state})
                    trace[-1]["weight"] = job.weight
                    continue
                level = _select_level(job, processor, now, mode, ready, processors)
                if level is None:
                    result["admission_infeasible"] += 1
                    terminal_state = None
                    if job.mandatory:
                        terminal_state = terminal_registry.transition(job.uid, "PRE_REJECT")
                    trace.append({"event": "dispatch_infeasible", "time": now,
                        "processor": processor, "uid": job.uid,
                        "image_uid": job.image_uid,
                        "source_image_id": job.source_image_id,
                        "region_index": job.region_index,
                        "mandatory": job.mandatory, "terminal_state": terminal_state})
                    trace[-1]["weight"] = job.weight
                    continue
                if now > state.temperature_time:
                    state.temperature = end_temperature(
                        state.temperature, 0.6, now - state.temperature_time
                    )
                    state.temperature_time = now
                state.running = Running(job, level, now + GPU_WCET[processor][level])
                trace.append({"event": "dispatch", "time": now,
                    "processor": processor, "uid": job.uid,
                    "mandatory": job.mandatory, "level": level + 1,
                    "voltage": BASELINE_VOLTAGE, "temperature": state.temperature})

    result.update(terminal_registry.finalize())
    result.update({"period_ms": period_ms, "lines": lines, "epochs": epochs,
                   "seed": seed, "scheduler": f"edf_{mode}"})
    return result, trace
