"""Resumable all-regions-mandatory schedulability phase boundary."""

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
OUT = ROOT / "experiments/checkpoints/v5_strict_all_regions_boundary"
RESULT = OUT / "cell_results.csv"
PERIODS = (50, 60, 70, 80, 90, 100, 150, 200, 250, 300)
LINES = (4, 5, 6, 7, 8, 9, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
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
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(RESULT)


def strict_stats(trace: list[dict], total_images: int) -> dict:
    terminal = {"complete", "mandatory_infeasible", "expired", "dispatch_infeasible"}
    rows = [x for x in trace if x.get("event") in terminal]
    rejected = sum(x["event"] == "mandatory_infeasible" for x in rows)
    admitted_late = sum(
        x["event"] in {"expired", "dispatch_infeasible"}
        or (x["event"] == "complete" and x.get("deadline_miss", False))
        for x in rows
    )
    completed_by_image: dict[str, int] = {}
    for x in rows:
        if x["event"] == "complete" and not x.get("deadline_miss", False):
            completed_by_image[x["image_uid"]] = completed_by_image.get(x["image_uid"], 0) + 1
    complete_images = sum(v == 4 for v in completed_by_image.values())
    completions = [x for x in trace if x.get("event") == "complete"]
    return {
        "pre_execution_rejections": rejected,
        "pre_execution_rejection_ratio": rejected / len(rows),
        "admitted_deadline_violations": admitted_late,
        "admitted_deadline_violation_ratio": admitted_late / max(1, len(rows) - rejected),
        "fully_accepted_images": complete_images,
        "full_image_on_time_acceptance": complete_images / total_images,
        "peak_modeled_temperature": max((x["temperature"] for x in completions), default=25.0),
        "mean_voltage": (sum(x["voltage"] for x in completions) / len(completions)
                         if completions else float("nan")),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    confusion = load_confusion(CONFUSION_PATH)
    rows: list[dict] = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(int(x["period_ms"]), int(x["lines"]), int(x["seed"])) for x in rows}
    total = len(PERIODS) * len(LINES) * len(SEEDS)
    started = time.time()
    for period in PERIODS:
        for lines in LINES:
            for seed in SEEDS:
                key = (period, lines, seed)
                if key in done:
                    continue
                t0 = time.time()
                summary, trace = run_continuous_hbtasp(
                    pool, period, lines, EPOCHS, seed,
                    budget_mode="assigned_level_sensitivity", network_mode="dynamic",
                    all_regions_mandatory=True,
                )
                scalar = score_historical_scalar(trace, summary["total_regions"])
                mask, _ = score_trace(trace, pool, confusion)
                row = {
                    **summary, **strict_stats(trace, lines * EPOCHS), **scalar, **mask,
                    "runtime_seconds": time.time() - t0,
                    "hardware_input_gpu0": "T4_measured_input",
                    "hardware_input_gpu1": "degraded_T4_simulation_input",
                    "simulation_host": "T4_execution_environment",
                }
                rows.append(row)
                write_rows(rows)
                done.add(key)
                cp = {
                    "status": "complete" if len(done) == total else "running",
                    "completed_cells": len(done), "total_cells": total,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds_this_run": time.time() - started,
                    "last_cell": {"period_ms": period, "lines": lines, "seed": seed},
                    "protocol": {"periods_ms": PERIODS, "lines": LINES,
                                 "seeds": SEEDS, "epochs": EPOCHS,
                                 "all_regions_mandatory": True,
                                 "release_mode": "periodic",
                                 "priority_overhead_ms": 0.0},
                    "sha256": {
                        "pool": sha256(POOL_PATH), "confusion": sha256(CONFUSION_PATH),
                        "hbtasp": sha256(ROOT / "experiments/initial_manuscript_hbtasp.py"),
                        "event_replay": sha256(ROOT / "experiments/initial_manuscript_event_replay.py"),
                        "spec": sha256(ROOT / "experiments/STRICT_AND_COOLING_EXPERIMENT_SPEC_V1.md"),
                        "results": sha256(RESULT),
                    },
                }
                (OUT / "checkpoint.json").write_text(json.dumps(cp, indent=2), encoding="utf-8")
                print(f"[{len(done)}/{total}] T={period} lines={lines} seed={seed} "
                      f"{row['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
