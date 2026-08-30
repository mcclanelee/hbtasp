"""Aligned 100--300 ms Top-1 priority quality/latency experiment."""

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
HISTOGRAM_POOL = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
MULTICUE_POOL = ROOT / "experiments/checkpoints/v5_multicue_priority/multicue_test_pool.json"
TIMING = ROOT / "experiments/checkpoints/v21_multicue_priority_aligned/priority_latency_4_6_8_10.json"
CONFUSION = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v21_multicue_priority_aligned"
RESULT = OUT / "cell_results.csv"

TREATMENTS = ("histogram_zero", "multicue_zero", "multicue_host_cpu_p99")
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EPOCHS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(rows):
    tmp = RESULT.with_suffix(".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(RESULT)


def mandatory_stats(trace):
    terminal = {"complete", "mandatory_infeasible", "optional_skipped", "expired", "dispatch_infeasible"}
    rows = [event for event in trace if event.get("event") in terminal and event.get("mandatory")]
    misses = sum(event["event"] != "complete" or event.get("deadline_miss", False) for event in rows)
    admitted_late = sum(event["event"] == "complete" and event.get("deadline_miss", False) for event in rows)
    return len(rows), misses, admitted_late


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    histogram = json.loads(HISTOGRAM_POOL.read_text(encoding="utf-8"))
    multicue = json.loads(MULTICUE_POOL.read_text(encoding="utf-8"))
    if [x["image_id"] for x in histogram] != [x["image_id"] for x in multicue]:
        raise RuntimeError("Histogram and multi-cue pools are not image-aligned")
    timing = json.loads(TIMING.read_text(encoding="utf-8"))
    local_p99 = {line: float(timing["results"][str(line)]["p99_ms"]) for line in LINES}
    pools = {"histogram_zero": histogram, "multicue_zero": multicue,
             "multicue_host_cpu_p99": multicue}
    confusion = load_confusion(CONFUSION)

    rows = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(x["treatment"], int(x["period_ms"]), int(x["lines"]), int(x["seed"])) for x in rows}
    total = len(TREATMENTS) * len(PERIODS) * len(LINES) * len(SEEDS)
    started = time.time()
    for treatment in TREATMENTS:
        for period in PERIODS:
            for lines in LINES:
                overhead = local_p99[lines] if treatment == "multicue_host_cpu_p99" else 0.0
                for seed in SEEDS:
                    key = (treatment, period, lines, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    summary, trace = run_continuous_hbtasp(
                        pools[treatment], period, lines, EPOCHS, seed,
                        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
                        priority_overhead_ms=overhead)
                    total_mandatory, missed_mandatory, admitted_late = mandatory_stats(trace)
                    scalar = score_historical_scalar(trace, summary["total_regions"])
                    mask, _ = score_trace(trace, pools[treatment], confusion)
                    rows.append({
                        "treatment": treatment, **summary,
                        "total_mandatory": total_mandatory,
                        "missed_mandatory": missed_mandatory,
                        "mandatory_service_failure": missed_mandatory / total_mandatory,
                        "admitted_mandatory_deadline_violations": admitted_late,
                        **scalar, **mask, "runtime_seconds": time.time() - t0,
                    })
                    write_rows(rows); done.add(key)
                    checkpoint = {
                        "status": "complete" if len(done) == total else "running",
                        "completed_cells": len(done), "total_cells": total,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds_this_run": time.time() - started,
                        "protocol": {"treatments": TREATMENTS, "periods_ms": PERIODS,
                                     "lines": LINES, "seeds": SEEDS, "epochs": EPOCHS,
                                     "host_cpu_p99_ms_by_lines": local_p99,
                                     "timing_evidence_class": "prototype_host_CPU_measurement",
                                     "evidence_scope": "host_preprocessing_separate_from_T4_GPU_profiles"},
                        "sha256": {"histogram_pool": sha256(HISTOGRAM_POOL),
                                   "multicue_pool": sha256(MULTICUE_POOL),
                                   "timing": sha256(TIMING), "confusion": sha256(CONFUSION),
                                   "results": sha256(RESULT)},
                    }
                    (OUT / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
                    print(f"[{len(done)}/{total}] {treatment} T={period} lines={lines} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
