"""Regression checks for the unified Overall baseline adapters."""

from __future__ import annotations

import json

from experiments.overall_baseline_adapters import run_esatd_l3, run_heat_l3
from experiments.run_v5_final_factorial import POOL_PATH


def _check(configuration, summary: dict, trace: list[dict], lines: int,
           epochs: int) -> None:
    released = lines * epochs * 4
    job_outcomes = [
        row for row in trace
        if row.get("event") in {"complete", "dispatch_infeasible"}
    ]
    mandatory_terminal = [
        row for row in job_outcomes
        if row.get("mandatory") and row.get("terminal_state")
    ]

    # Every released job is either completed or explicitly rejected after both
    # cores are already past its deadline. A late dispatched job still executes
    # and therefore remains visible as LATE_COMPLETE.
    assert summary["completed"] + summary["admission_infeasible"] == released, configuration
    assert len(job_outcomes) == released, configuration
    assert summary["mandatory_released"] == lines * epochs, configuration
    assert len(mandatory_terminal) == lines * epochs, configuration
    assert summary["mandatory_terminal_residual"] == 0, configuration
    assert summary["mandatory_unique_terminal_uids"] == lines * epochs, configuration
    assert summary["mandatory_wait_expire"] == 0, configuration
    assert (
        summary["mandatory_pre_reject"]
        + summary["mandatory_late_complete"]
        + summary["mandatory_on_time"]
        == lines * epochs
    ), configuration
    assert len({row["uid"] for row in job_outcomes}) == released, configuration
    completed_outcomes = [row for row in job_outcomes if row["event"] == "complete"]
    assert sum(bool(row["deadline_miss"]) for row in completed_outcomes) == summary[
        "deadline_misses"
    ], configuration
    # Tight load must exercise the optional deadline-credit path; otherwise a
    # regression that marks only mandatory jobs late could pass unnoticed.
    assert any(
        row.get("deadline_miss") and not row.get("mandatory")
        for row in completed_outcomes
    ), configuration


def main() -> None:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    lines, epochs, seed = 10, 12, 101
    for name, runner in (("ESATD-L3", run_esatd_l3), ("HEAT-L3", run_heat_l3)):
        summary, trace = runner(pool, 100, lines, epochs, seed)
        _check(name, summary, trace, lines, epochs)
    print("OVERALL BASELINE ADAPTER TEST PASSED")


if __name__ == "__main__":
    main()
