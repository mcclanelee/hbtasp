"""Wide-range paired audit of the manuscript productive-cooling branch.

The experiment changes no scheduling rule.  It extends the period range and
records where the branch acts before testing whether its local voltage and
temperature effects propagate to system-level service and perception metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.v18_instrumented_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import load_confusion, score_historical_scalar, score_trace
from experiments.run_v5_productive_cooling_ablation import trace_stats

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "experiments/checkpoints/initial_histogram_pool_v1/initial_histogram_test_pool.json"
CONFUSION = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v18_wide_cooling_mechanism"
RESULT = OUT / "cell_results.csv"
PERIODS = (40, 50, 60, 80, 100, 150, 200, 300, 500, 750, 1000)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
COOLING = (False, True)
EPOCHS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = RESULT.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(RESULT)


def mechanism_stats(trace: list[dict]) -> dict:
    dispatches = [x for x in trace if x.get("event") == "dispatch"]
    completions = [x for x in trace if x.get("event") == "complete"]
    branch = [x for x in dispatches if x.get("cooling_branch", False)]
    branch_complete = [x for x in completions if x.get("cooling_branch", False)]
    local_deltas = [x["planned_end_temperature"] - x["planned_start_temperature"]
                    for x in branch]
    return {
        "dispatches": len(dispatches),
        "cooling_branch_dispatches": len(branch),
        "cooling_branch_rate": len(branch) / max(1, len(dispatches)),
        "cooling_locally_decreasing_dispatches": sum(d < 0 for d in local_deltas),
        "cooling_locally_decreasing_rate": sum(d < 0 for d in local_deltas) / max(1, len(branch)),
        "cooling_mean_planned_delta_c": sum(local_deltas) / max(1, len(local_deltas)),
        "cooling_total_iit_celsius_seconds": sum(
            x.get("iit_celsius_seconds", 0.0) for x in branch_complete),
        "total_iit_celsius_seconds": sum(
            x.get("iit_celsius_seconds", 0.0) for x in completions),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    confusion = load_confusion(CONFUSION)
    rows = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(x["cooling_enabled"].lower() == "true", int(x["period_ms"]),
             int(x["lines"]), int(x["seed"])) for x in rows}
    total = len(COOLING) * len(PERIODS) * len(LINES) * len(SEEDS)
    started = time.time()
    for enabled in COOLING:
        for period in PERIODS:
            for lines in LINES:
                for seed in SEEDS:
                    key = (enabled, period, lines, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    summary, trace = run_continuous_hbtasp(
                        pool, period, lines, EPOCHS, seed,
                        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
                        enable_productive_cooling=enabled,
                    )
                    row = {"cooling_enabled": enabled, **summary,
                           **trace_stats(trace), **mechanism_stats(trace),
                           **score_historical_scalar(trace, summary["total_regions"]),
                           **score_trace(trace, pool, confusion)[0],
                           "runtime_seconds": time.time() - t0}
                    rows.append(row); write_rows(rows); done.add(key)
                    checkpoint = {
                        "status": "complete" if len(done) == total else "running",
                        "completed_cells": len(done), "total_cells": total,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds_this_run": time.time() - started,
                        "protocol": {"paired_branch_settings": COOLING,
                                     "periods_ms": PERIODS, "lines": LINES,
                                     "seeds": SEEDS, "epochs": EPOCHS,
                                     "algorithm": "initial-manuscript HBTASP unchanged"},
                        "sha256": {"pool": sha256(POOL), "confusion": sha256(CONFUSION),
                                   "runner": sha256(Path(__file__)),
                                   "instrumented_event_replay": sha256(
                                       ROOT / "experiments/v18_instrumented_event_replay.py"),
                                   "results": sha256(RESULT)},
                    }
                    (OUT / "checkpoint.json").write_text(
                        json.dumps(checkpoint, indent=2), encoding="utf-8")
                    print(f"[{len(done)}/{total}] cooling={enabled} T={period} "
                          f"lines={lines} seed={seed} {row['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
