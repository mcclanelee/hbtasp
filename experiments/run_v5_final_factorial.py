"""Authoritative 2x2 rerun after the selected-level-budget state fix."""

from __future__ import annotations

import csv, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

from experiments.initial_edf_event_replay import run_edf
from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import load_confusion, score_historical_scalar, score_trace

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/initial_histogram_pool_v1/initial_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v5_final_factorial_selected_budget"
RESULT = OUT / "cell_results.csv"
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EPOCHS = 1000
CONFIGS = ("EDF-FixedL3", "EDF-Dynamic-Reservation", "HBTASP-FixedL3", "HBTASP-Dynamic")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mandatory_stats(trace):
    terminal = {"complete", "mandatory_infeasible", "optional_skipped", "expired", "dispatch_infeasible"}
    rows = [x for x in trace if x.get("event") in terminal and x.get("mandatory")]
    misses = sum(x["event"] != "complete" or x.get("deadline_miss", False) for x in rows)
    return len(rows), misses


def run_cell(config, pool, period, lines, seed):
    if config == "EDF-FixedL3":
        summary, trace = run_edf(pool, period, lines, EPOCHS, seed, "fixed_l3")
    elif config == "EDF-Dynamic-Reservation":
        summary, trace = run_edf(pool, period, lines, EPOCHS, seed, "dynamic_reservation")
    elif config == "HBTASP-FixedL3":
        summary, trace = run_continuous_hbtasp(
            pool, period, lines, EPOCHS, seed,
            budget_mode="manuscript_l5", network_mode="fixed_l3")
    elif config == "HBTASP-Dynamic":
        summary, trace = run_continuous_hbtasp(
            pool, period, lines, EPOCHS, seed,
            budget_mode="assigned_level_sensitivity", network_mode="dynamic")
    else:
        raise ValueError(config)
    total, missed = mandatory_stats(trace)
    return summary, trace, {"total_mandatory": total, "missed_mandatory": missed,
                            "mandatory_dmr": missed / total}


def write_rows(rows):
    tmp = RESULT.with_suffix(".tmp")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(RESULT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    confusion = load_confusion(CONFUSION_PATH)
    rows = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(x["configuration"], int(x["period_ms"]), int(x["lines"]), int(x["seed"])) for x in rows}
    total_cells = len(CONFIGS) * len(PERIODS) * len(LINES) * len(SEEDS)
    started = time.time()
    for config in CONFIGS:
        for period in PERIODS:
            for lines in LINES:
                for seed in SEEDS:
                    key = (config, period, lines, seed)
                    if key in done:
                        continue
                    t0 = time.time()
                    summary, trace, mandatory = run_cell(config, pool, period, lines, seed)
                    scalar = score_historical_scalar(trace, summary["total_regions"])
                    mask, _ = score_trace(trace, pool, confusion)
                    row = {"configuration": config, **summary, **mandatory, **scalar, **mask,
                           "runtime_seconds": time.time() - t0,
                           "scalar_quality_definition": "historical_image_averaged_path_dice",
                           "mask_metric_status": "confusion_replay_zero_credit"}
                    rows.append(row); write_rows(rows); done.add(key)
                    checkpoint = {
                        "status": "complete" if len(done) == total_cells else "running",
                        "completed_cells": len(done), "total_cells": total_cells,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds_this_run": time.time() - started,
                        "last_cell": {"configuration": config, "period_ms": period,
                                      "lines": lines, "seed": seed},
                        "protocol": {"periods": PERIODS, "lines": LINES, "seeds": SEEDS,
                                     "epochs": EPOCHS,
                                     "budget": "Stage-I selected-level from construction time",
                                     "release_mode": "periodic", "mandatory_tie": "mandatory first"},
                        "sha256": {"pool": sha256(POOL_PATH), "confusion": sha256(CONFUSION_PATH),
                                   "hbtasp": sha256(ROOT / "experiments/initial_manuscript_hbtasp.py"),
                                   "event_replay": sha256(ROOT / "experiments/initial_manuscript_event_replay.py"),
                                   "edf": sha256(ROOT / "experiments/initial_edf_event_replay.py"),
                                   "data_authority": sha256(ROOT / "experiments/DATA_AUTHORITY_V5.md"),
                                   "results": sha256(RESULT)}}
                    (OUT / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
                    print(f"[{len(done)}/{total_cells}] {config} T={period} lines={lines} seed={seed} "
                          f"{row['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
