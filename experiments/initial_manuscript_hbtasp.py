"""Faithful primitives for the initially submitted HBTASP algorithm.

This module is deliberately isolated from the corrected/revised schedulers.
Its authority is INITIAL_MANUSCRIPT_ALGORITHM_LOCK_V3.md.  Times are seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


VOLTAGES = (0.60, 0.70, 0.80, 0.85, 0.88)
BASELINE_VOLTAGE = 0.80
GPU_WCET = (
    (0.0063, 0.0090, 0.0123, 0.0148, 0.0201),
    (0.0130, 0.0182, 0.0251, 0.0295, 0.0400),
)
DICE = (0.5790, 0.6183, 0.6441, 0.6730, 0.6884)
THERMAL_A = 0.1
THERMAL_ALPHA = 41.01
THERMAL_GAMMA = 512.635
THERMAL_B = 0.5058
TMAX = 60.0
TAMB = 25.0


def local_edf_key(item: StageOneAssignment):
    """EDF with stable mandatory-first tie breaking from two-queue admission."""
    return (item.job.deadline, 0 if item.job.mandatory else 1, item.job.uid)


@dataclass(frozen=True)
class RegionJob:
    uid: str
    image_uid: str
    release: float
    deadline: float
    weight: float
    mandatory: bool
    source_image_id: str = ""
    region_index: int = -1


@dataclass(frozen=True)
class StageOneAssignment:
    job: RegionJob
    processor: int
    initial_level: int
    mu: float


@dataclass(frozen=True)
class DispatchDecision:
    assignment: StageOneAssignment
    level: int
    voltage_index: int
    duration: float
    start_temperature: float
    end_temperature: float
    heavy: bool
    cooling: bool


@dataclass(frozen=True)
class OptionalAssignmentAudit:
    lp_status: int
    lp_objective: float
    fractional_variables: int
    assigned: int
    skipped: tuple[str, ...]
    downgrade_steps: int
    upgrade_steps: int


def steady_temperature(voltage: float) -> float:
    return THERMAL_A * (THERMAL_ALPHA + THERMAL_GAMMA * voltage**3) / THERMAL_B


def end_temperature(start: float, voltage: float, duration: float) -> float:
    tss = steady_temperature(voltage)
    return tss - (tss - start) * exp(-THERMAL_B * duration)


def duration_at_voltage(processor: int, level: int, voltage_index: int) -> float:
    """Equation (4): WCET scales inversely with voltage from the 0.8-V table."""
    return GPU_WCET[processor][level] * BASELINE_VOLTAGE / VOLTAGES[voltage_index]


def manuscript_budget(processor: int) -> float:
    """Primary interpretation of manuscript mu: L5 WCET at baseline voltage."""
    return GPU_WCET[processor][4]


def project_edf_finish_times(
    assignments: Sequence[StageOneAssignment],
    available_at: float = 0.0,
) -> list[tuple[StageOneAssignment, float, float]]:
    """Project non-preemptive baseline execution in stable local EDF order."""
    ordered = sorted(assignments, key=local_edf_key)
    projected: list[tuple[StageOneAssignment, float, float]] = []
    available = available_at
    for item in ordered:
        start = max(available, item.job.release)
        duration = GPU_WCET[item.processor][item.initial_level]
        finish = start + duration
        projected.append((item, start, finish))
        available = finish
    return projected


def queue_is_deadline_feasible(
    assignments: Sequence[StageOneAssignment], available_at: float = 0.0
) -> bool:
    return all(
        finish <= item.job.deadline + 1e-12
        for item, _start, finish in project_edf_finish_times(assignments, available_at)
    )


def assign_mandatory(
    jobs: Sequence[RegionJob],
    processor_queues: Sequence[Sequence[StageOneAssignment]] | None = None,
    processor_available: Sequence[float] | None = None,
    mandatory_level: int = 4,
    budget_level: int = 4,
) -> tuple[list[list[StageOneAssignment]], list[RegionJob]]:
    """Algorithm-1 mandatory pass without inventing mandatory eviction.

    Feasible processors are selected by the manuscript's minimum projected
    load rule. If no processor is feasible, the job is explicitly returned as
    infeasible; mandatory work is never downgraded, skipped, or evicted.
    """
    queues = [list(q) for q in (processor_queues or ((), ()))]
    available = list(processor_available or (0.0, 0.0))
    infeasible: list[RegionJob] = []
    mandatory_jobs = sorted(
        (job for job in jobs if job.mandatory),
        key=lambda job: (job.deadline, job.uid),
    )
    for job in mandatory_jobs:
        candidates: list[tuple[float, float, int, StageOneAssignment]] = []
        for processor in range(len(GPU_WCET)):
            item = StageOneAssignment(
                job, processor, mandatory_level, GPU_WCET[processor][budget_level]
            )
            trial = queues[processor] + [item]
            if queue_is_deadline_feasible(trial, available[processor]):
                projected = project_edf_finish_times(trial, available[processor])
                last_finish = max((finish for _, _, finish in projected), default=0.0)
                total_load = sum(
                    GPU_WCET[x.processor][x.initial_level] for x in trial
                )
                candidates.append((total_load, last_finish, processor, item))
        if not candidates:
            infeasible.append(job)
            continue
        _load, _finish, processor, item = min(candidates)
        queues[processor].append(item)
        queues[processor].sort(key=local_edf_key)
    return queues, infeasible


def _optional_ratio(job: RegionJob, processor: int, low: int, high: int) -> float:
    added = GPU_WCET[processor][high] - GPU_WCET[processor][low]
    if added <= 0:
        return float("-inf")
    return job.weight * (DICE[high] - DICE[low]) / added


def assign_optional_manuscript(
    jobs: Sequence[RegionJob],
    processor_queues: Sequence[Sequence[StageOneAssignment]],
    processor_available: Sequence[float] | None = None,
    allowed_levels: Sequence[int] = (0, 1, 2, 3, 4),
    budget_level: int = 4,
    enforce_initial_mandatory_l5: bool = True,
    budget_from_selected_level: bool = False,
) -> tuple[list[list[StageOneAssignment]], OptionalAssignmentAudit]:
    """Algorithm-2 LP relaxation, stable rounding, downgrade, and upgrade.

    This uses the finite-batch cumulative constraints printed in the initial
    manuscript. Processor-local recurrence is rechecked after every discrete
    choice, which is an allowed dimensional/parallelism repair in the lock.
    Optional work for which no feasible rounded assignment exists is skipped.
    """
    queues = [list(q) for q in processor_queues]
    available = list(processor_available or (0.0, 0.0))
    optional = sorted(
        (job for job in jobs if not job.mandatory),
        key=lambda job: (job.deadline, job.uid),
    )
    if not optional:
        return queues, OptionalAssignmentAudit(0, 0.0, 0, 0, (), 0, 0)

    variable_keys = [
        (i, processor, level)
        for i in range(len(optional))
        for processor in range(len(GPU_WCET))
        for level in allowed_levels
    ]
    c = np.array([
        -optional[i].weight * DICE[level]
        for i, _processor, level in variable_keys
    ])
    rows: list[np.ndarray] = []
    bounds: list[float] = []

    # At most one processor-level choice for every optional region.
    for i in range(len(optional)):
        row = np.zeros(len(variable_keys))
        for q, (ii, _processor, _level) in enumerate(variable_keys):
            if ii == i:
                row[q] = 1.0
        rows.append(row)
        bounds.append(1.0)

    # Manuscript cumulative EDF constraint, repaired to be processor-specific.
    for processor in range(len(GPU_WCET)):
        base = list(queues[processor])
        for prefix, current in enumerate(optional):
            row = np.zeros(len(variable_keys))
            for q, (i, p, level) in enumerate(variable_keys):
                if p == processor and i <= prefix:
                    row[q] = GPU_WCET[p][level]
            base_work = sum(
                GPU_WCET[x.processor][x.initial_level]
                for x in base if x.job.deadline <= current.deadline
            )
            origin = min(
                [current.release]
                + [x.job.release for x in base if x.job.deadline <= current.deadline]
                + [x.release for x in optional[: prefix + 1]]
            )
            rows.append(row)
            occupied = max(0.0, available[processor] - origin)
            bounds.append(max(0.0, current.deadline - origin - occupied - base_work))

    result = linprog(
        c,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(bounds),
        bounds=[(0.0, 1.0)] * len(variable_keys),
        method="highs",
    )
    if not result.success:
        return queues, OptionalAssignmentAudit(
            int(result.status), float("nan"), 0, 0,
            tuple(job.uid for job in optional), 0, 0,
        )

    fractional = sum(1 for value in result.x if 1e-9 < value < 1.0 - 1e-9)
    skipped: list[str] = []
    downgrade_steps = 0

    for i, job in enumerate(optional):
        positive = [
            (result.x[q], processor, level)
            for q, (ii, processor, level) in enumerate(variable_keys)
            if ii == i and result.x[q] > 1e-9
        ]
        # Integral choices are retained. Fractional choices are considered by
        # the manuscript load-balance rule, then by descending LP mass.
        candidates = sorted(
            positive,
            key=lambda value: (
                sum(GPU_WCET[x.processor][x.initial_level]
                    for x in queues[value[1]]) + GPU_WCET[value[1]][value[2]],
                -value[0], value[1], value[2],
            ),
        )
        selected = None
        for _mass, processor, level in candidates:
            item_budget = GPU_WCET[processor][
                level if budget_from_selected_level else budget_level
            ]
            item = StageOneAssignment(job, processor, level, item_budget)
            if queue_is_deadline_feasible(
                queues[processor] + [item], available[processor]
            ):
                selected = item
                break
            # Algorithm 2 downgrades until the local EDF queue is feasible.
            for lower in range(level - 1, -1, -1):
                downgraded_budget = GPU_WCET[processor][
                    lower if budget_from_selected_level else budget_level
                ]
                downgraded = StageOneAssignment(
                    job, processor, lower, downgraded_budget
                )
                if queue_is_deadline_feasible(
                    queues[processor] + [downgraded], available[processor]
                ):
                    selected = downgraded
                    downgrade_steps += level - lower
                    break
            if selected is not None:
                break
        if selected is None:
            skipped.append(job.uid)
            continue
        queues[selected.processor].append(selected)
        queues[selected.processor].sort(key=local_edf_key)

    # Manuscript upgrade pass: maximum weighted accuracy gain per added time.
    upgrade_steps = 0
    while True:
        candidates = []
        for processor, queue in enumerate(queues):
            for index, item in enumerate(queue):
                if item.job.mandatory or item.initial_level >= max(allowed_levels):
                    continue
                next_levels = [level for level in allowed_levels if level > item.initial_level]
                if not next_levels:
                    continue
                next_level = min(next_levels)
                ratio = _optional_ratio(
                    item.job, processor, item.initial_level, next_level
                )
                candidates.append((-ratio, item.job.uid, processor, index))
        changed = False
        for _negative_ratio, _uid, processor, index in sorted(candidates):
            old = queues[processor][index]
            upgraded_level = min(
                level for level in allowed_levels if level > old.initial_level
            )
            upgraded = StageOneAssignment(
                old.job, processor, upgraded_level,
                (GPU_WCET[processor][upgraded_level]
                 if budget_from_selected_level else old.mu),
            )
            trial = list(queues[processor])
            trial[index] = upgraded
            if queue_is_deadline_feasible(trial, available[processor]):
                queues[processor] = sorted(
                    trial, key=local_edf_key
                )
                upgrade_steps += 1
                changed = True
                break
        if not changed:
            break

    if enforce_initial_mandatory_l5:
        for queue in queues:
            validate_local_queue(queue)
    return queues, OptionalAssignmentAudit(
        int(result.status), float(-result.fun), fractional,
        sum(len(q) for q in queues) - sum(len(q) for q in processor_queues),
        tuple(skipped), downgrade_steps, upgrade_steps,
    )


def classify_heavy(job: RegionJob, remaining: Sequence[StageOneAssignment]) -> bool:
    if not remaining:
        return False
    mean_weight = sum(item.job.weight for item in remaining) / len(remaining)
    return job.weight > mean_weight


def _feasible(
    assignment: StageOneAssignment,
    level: int,
    voltage_index: int,
    temperature: float,
) -> tuple[float, float] | None:
    duration = duration_at_voltage(assignment.processor, level, voltage_index)
    if duration > assignment.mu + 1e-12:
        return None
    end = end_temperature(temperature, VOLTAGES[voltage_index], duration)
    if end > TMAX + 1e-10:
        return None
    return duration, end


def select_stage_two(
    assignment: StageOneAssignment,
    remaining_queue: Sequence[StageOneAssignment],
    temperature: float,
    fixed_level: int | None = None,
    enable_productive_cooling: bool = True,
    fixed_voltage_index: int | None = None,
    enforce_thermal_feasibility: bool = True,
) -> DispatchDecision | None:
    """Apply Algorithm 3 literally to the current EDF-ordered local queue."""
    if not remaining_queue or remaining_queue[0].job.uid != assignment.job.uid:
        raise ValueError("assignment must be the head of remaining_queue")

    heavy = classify_heavy(assignment.job, remaining_queue)
    next_heavy = (
        len(remaining_queue) > 1
        and classify_heavy(remaining_queue[1].job, remaining_queue)
    )
    # The ablation flag disables only the manuscript's productive-cooling
    # branch.  True is the original Algorithm-3 behavior.
    cooling = enable_productive_cooling and (not heavy) and next_heavy

    # The manuscript fixes mandatory work at L5 irrespective of classification.
    levels: Iterable[int] = ((fixed_level,) if fixed_level is not None else
                             ((4,) if assignment.job.mandatory else range(4, -1, -1)))

    if fixed_voltage_index is not None:
        if fixed_voltage_index not in range(len(VOLTAGES)):
            raise ValueError("fixed_voltage_index is outside the voltage set")
        for level in levels:
            duration = duration_at_voltage(
                assignment.processor, level, fixed_voltage_index
            )
            if duration > assignment.mu + 1e-12:
                continue
            end = end_temperature(
                temperature, VOLTAGES[fixed_voltage_index], duration
            )
            if enforce_thermal_feasibility and end > TMAX + 1e-10:
                continue
            return DispatchDecision(
                assignment, level, fixed_voltage_index, duration,
                temperature, end, heavy, False,
            )
        return None

    if cooling:
        # Lowest discrete timing-feasible voltage; then largest feasible level.
        for voltage_index in range(len(VOLTAGES)):
            for level in levels:
                feasible = _feasible(assignment, level, voltage_index, temperature)
                if feasible is not None:
                    duration, end = feasible
                    return DispatchDecision(
                        assignment, level, voltage_index, duration,
                        temperature, end, heavy, True,
                    )
        return None

    # Heavy strategy: highest voltage first, largest feasible level at that V.
    for voltage_index in range(len(VOLTAGES) - 1, -1, -1):
        for level in levels:
            feasible = _feasible(assignment, level, voltage_index, temperature)
            if feasible is not None:
                duration, end = feasible
                return DispatchDecision(
                    assignment, level, voltage_index, duration,
                    temperature, end, heavy, False,
                )
    return None


def validate_local_queue(assignments: Sequence[StageOneAssignment]) -> None:
    if list(assignments) != sorted(assignments, key=local_edf_key):
        raise ValueError("processor-local queue is not in stable EDF order")
    seen: set[str] = set()
    for item in assignments:
        if item.job.uid in seen:
            raise ValueError(f"duplicate assignment: {item.job.uid}")
        seen.add(item.job.uid)
        if item.job.mandatory and item.initial_level != 4:
            raise ValueError("mandatory Stage-I assignment must use L5")
        expected_mu = manuscript_budget(item.processor)
        if abs(item.mu - expected_mu) > 1e-12:
            raise ValueError("primary manuscript budget must equal L5@0.8V")
