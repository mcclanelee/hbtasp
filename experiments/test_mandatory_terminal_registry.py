"""Fast regression checks for kernel-owned mandatory terminal states."""

from __future__ import annotations

import json

from experiments.initial_edf_event_replay import run_edf
from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.run_v5_final_factorial import POOL_PATH


def validate(summary: dict, trace: list[dict]) -> None:
    events = [
        event for event in trace
        if event.get("mandatory") and event.get("terminal_state") is not None
    ]
    assert summary["mandatory_terminal_residual"] == 0
    assert len(events) == summary["mandatory_released"]
    assert len(events) == summary["mandatory_unique_terminal_uids"]
    assert len({event["uid"] for event in events}) == len(events)
    assert sum(summary[key] for key in (
        "mandatory_pre_reject", "mandatory_wait_expire",
        "mandatory_late_complete", "mandatory_on_time",
    )) == summary["mandatory_released"]


def main() -> None:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    cases = (
        run_edf(pool, 100, 4, 20, 101, "fixed_l3"),
        run_edf(pool, 100, 4, 20, 101, "dynamic_reservation"),
        run_continuous_hbtasp(
            pool, 100, 4, 20, 101,
            budget_mode="manuscript_l5", network_mode="fixed_l3",
        ),
        run_continuous_hbtasp(
            pool, 100, 4, 20, 101,
            budget_mode="assigned_level_sensitivity", network_mode="dynamic",
        ),
    )
    for summary, trace in cases:
        validate(summary, trace)
    print("MANDATORY TERMINAL REGISTRY TEST PASSED")


if __name__ == "__main__":
    main()
