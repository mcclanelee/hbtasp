"""Resumable same-kernel productive-cooling ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import load_confusion, score_historical_scalar, score_trace

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/initial_histogram_pool_v1/initial_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v5_productive_cooling_ablation"
RESULT = OUT / "cell_results.csv"
PERIODS = (60, 80, 100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
COOLING = (False, True)
EPOCHS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(rows: list[dict]) -> None:
    tmp = RESULT.with_suffix(".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    # On Windows a concurrent read can briefly prevent atomic replacement.
    # Retry the rename without recomputing or discarding the completed cell.
    for attempt in range(20):
        try:
            tmp.replace(RESULT)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def trace_stats(trace: list[dict]) -> dict:
    terminal = {"complete", "mandatory_infeasible", "optional_skipped", "expired", "dispatch_infeasible"}
    mandatory = [x for x in trace if x.get("event") in terminal and x.get("mandatory")]
    rejected = sum(x["event"] == "mandatory_infeasible" for x in mandatory)
    admitted_late = sum(
        x["event"] in {"expired", "dispatch_infeasible"}
        or (x["event"] == "complete" and x.get("deadline_miss", False))
        for x in mandatory
    )
    complete = [x for x in trace if x.get("event") == "complete"]
    counts = {level: sum(x["level"] == level for x in complete) for level in range(1, 6)}
    return {
        "total_mandatory": len(mandatory),
        "mandatory_pre_execution_rejections": rejected,
        "mandatory_rejection_ratio": rejected / len(mandatory),
        "admitted_mandatory_deadline_violations": admitted_late,
        "admitted_mandatory_violation_ratio": admitted_late / max(1, len(mandatory) - rejected),
        "peak_modeled_temperature": max((x["temperature"] for x in complete), default=25.0),
        "mean_voltage": (sum(x["voltage"] for x in complete) / len(complete)
                         if complete else float("nan")),
        **{f"completed_l{level}": counts[level] for level in range(1, 6)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    confusion = load_confusion(CONFUSION_PATH)
    rows: list[dict] = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(x["cooling_enabled"].lower() == "true", int(x["period_ms"]),
             int(x["lines"]), int(x["seed"])) for x in rows}
    total = len(COOLING) * len(PERIODS) * len(LINES) * len(SEEDS)
    started = time.time()
    for cooling in COOLING:
        for period in PERIODS:
            for lines in LINES:
                for seed in SEEDS:
                    key = (cooling, period, lines, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    summary, trace = run_continuous_hbtasp(
                        pool, period, lines, EPOCHS, seed,
                        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
                        enable_productive_cooling=cooling,
                    )
                    scalar = score_historical_scalar(trace, summary["total_regions"])
                    mask, _ = score_trace(trace, pool, confusion)
                    row = {
                        "cooling_enabled": cooling, **summary, **trace_stats(trace),
                        **scalar, **mask, "runtime_seconds": time.time() - t0,
                        "hardware_input_gpu0": "T4_measured_input",
                        "hardware_input_gpu1": "power_capped_T4_measured_input",
                        "simulation_host": "T4_execution_environment",
                    }
                    rows.append(row); write_rows(rows); done.add(key)
                    cp = {
                        "status": "complete" if len(done) == total else "running",
                        "completed_cells": len(done), "total_cells": total,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds_this_run": time.time() - started,
                        "last_cell": {"cooling_enabled": cooling, "period_ms": period,
                                      "lines": lines, "seed": seed},
                        "protocol": {"cooling": COOLING, "periods_ms": PERIODS,
                                     "lines": LINES, "seeds": SEEDS, "epochs": EPOCHS,
                                     "release_mode": "periodic", "priority_overhead_ms": 0.0},
                        "sha256": {
                            "pool": sha256(POOL_PATH), "confusion": sha256(CONFUSION_PATH),
                            "hbtasp": sha256(ROOT / "experiments/initial_manuscript_hbtasp.py"),
                            "event_replay": sha256(ROOT / "experiments/initial_manuscript_event_replay.py"),
                            "results": sha256(RESULT),
                        },
                        "protocol_source": "embedded_checkpoint_protocol",
                    }
                    (OUT / "checkpoint.json").write_text(json.dumps(cp, indent=2), encoding="utf-8")
                    print(f"[{len(done)}/{total}] cooling={cooling} T={period} "
                          f"lines={lines} seed={seed} {row['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
