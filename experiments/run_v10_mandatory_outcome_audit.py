"""Replay the frozen V8 2x2 protocol and expose terminal-service semantics.

This audit does not modify either scheduler.  It adds a mutually exclusive
decomposition of every released mandatory region and an executed-level audit
to the existing per-cell output.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone

from experiments import run_v5_final_factorial as factorial
from experiments.initial_r2_8_metrics import load_confusion, score_historical_scalar, score_trace


OUT = factorial.ROOT / "experiments/checkpoints/v10_mandatory_outcome_audit"
RESULT = OUT / "cell_results.csv"
SOURCE_POOL = factorial.ROOT / (
    "experiments/checkpoints/v8_calibrated_final_factorial/"
    "calibrated_histogram_test_pool.json"
)


def terminal_audit(trace: list[dict]) -> dict:
    terminal = {"complete", "mandatory_infeasible", "optional_skipped",
                "expired", "dispatch_infeasible"}
    rows = [event for event in trace if event.get("event") in terminal]
    mandatory = [event for event in rows if event.get("mandatory")]

    allocation_rejected = sum(e["event"] == "mandatory_infeasible" for e in mandatory)
    dispatch_infeasible = sum(e["event"] == "dispatch_infeasible" for e in mandatory)
    expired = sum(e["event"] == "expired" for e in mandatory)
    completed_late = sum(e["event"] == "complete" and e.get("deadline_miss", False)
                         for e in mandatory)
    completed_on_time = sum(e["event"] == "complete" and not e.get("deadline_miss", False)
                            for e in mandatory)
    completed = completed_on_time + completed_late
    released = len(mandatory)
    service_failures = allocation_rejected + dispatch_infeasible + expired + completed_late
    if released != completed_on_time + service_failures:
        raise RuntimeError("mandatory terminal outcomes are not exhaustive")

    completed_all = [e for e in rows if e["event"] == "complete"]
    completed_mandatory = [e for e in completed_all if e.get("mandatory")]
    completed_optional = [e for e in completed_all if not e.get("mandatory")]
    all_levels = Counter(int(e["level"]) for e in completed_all)
    mandatory_levels = Counter(int(e["level"]) for e in completed_mandatory)
    optional_levels = Counter(int(e["level"]) for e in completed_optional)

    result = {
        "mandatory_released": released,
        "mandatory_completed_on_time": completed_on_time,
        "mandatory_completed_late": completed_late,
        "mandatory_expired": expired,
        "mandatory_allocation_rejected": allocation_rejected,
        "mandatory_dispatch_infeasible": dispatch_infeasible,
        "mandatory_preexecution_failure": allocation_rejected + dispatch_infeasible,
        "mandatory_service_failures": service_failures,
        "mandatory_on_time_service_rate": completed_on_time / released,
        "mandatory_service_failure_rate": service_failures / released,
        "mandatory_completed_job_dmr": completed_late / completed if completed else 0.0,
        "mandatory_admitted_failure_rate": (
            (expired + completed_late) /
            (released - allocation_rejected - dispatch_infeasible)
            if released > allocation_rejected + dispatch_infeasible else 0.0
        ),
        "all_completed_regions": len(completed_all),
        "mandatory_completed_regions": len(completed_mandatory),
        "optional_completed_regions": len(completed_optional),
    }
    for level in range(1, 6):
        result[f"completed_l{level}"] = all_levels[level]
        result[f"mandatory_completed_l{level}"] = mandatory_levels[level]
        result[f"optional_completed_l{level}"] = optional_levels[level]
    return result


def write_rows(rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = RESULT.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(SOURCE_POOL.read_text(encoding="utf-8"))
    confusion = load_confusion(factorial.CONFUSION_PATH)
    rows: list[dict] = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(r["configuration"], int(r["period_ms"]), int(r["lines"]), int(r["seed"]))
            for r in rows}
    total = len(factorial.CONFIGS) * len(factorial.PERIODS) * len(factorial.LINES) * len(factorial.SEEDS)
    started = time.time()

    for config in factorial.CONFIGS:
        for period in factorial.PERIODS:
            for lines in factorial.LINES:
                for seed in factorial.SEEDS:
                    key = (config, period, lines, seed)
                    if key in done:
                        continue
                    begin = time.time()
                    summary, trace, legacy = factorial.run_cell(config, pool, period, lines, seed)
                    scalar = score_historical_scalar(trace, summary["total_regions"])
                    mask, _ = score_trace(trace, pool, confusion)
                    audit = terminal_audit(trace)
                    row = {"configuration": config, **summary, **legacy, **audit, **scalar, **mask,
                           "period_ms": period, "lines": lines, "epochs": factorial.EPOCHS,
                           "seed": seed, "runtime_seconds": time.time() - begin,
                           "protocol": "frozen_v8_calibrated_2x2_semantic_replay"}
                    if abs(float(row["mandatory_dmr"]) - audit["mandatory_service_failure_rate"]) > 1e-12:
                        raise RuntimeError("legacy mandatory_dmr is not the service-failure rate")
                    rows.append(row)
                    write_rows(rows)
                    done.add(key)
                    checkpoint = {
                        "status": "complete" if len(done) == total else "running",
                        "completed_cells": len(done), "total_cells": total,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds_this_run": time.time() - started,
                        "semantic_invariant": "legacy mandatory_dmr == mandatory_service_failure_rate",
                        "decision_mechanism": "unchanged frozen V8 schedulers",
                        "source_pool_sha256": factorial.sha256(SOURCE_POOL),
                        "result_sha256": factorial.sha256(RESULT),
                    }
                    (OUT / "checkpoint.json").write_text(
                        json.dumps(checkpoint, indent=2), encoding="utf-8")
                    print(f"[{len(done)}/{total}] {config} T={period} lines={lines} seed={seed} "
                          f"{row['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
