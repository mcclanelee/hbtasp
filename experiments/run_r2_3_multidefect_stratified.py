"""R2.3 diagnostic: zero-credit metrics stratified by defective-region count.

The scheduler and mask scorer are unchanged.  The replay uses the same 1,000
cycles, periods, line counts, and seeds as the headline dynamic comparison.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from experiments.initial_edf_event_replay import run_edf
from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import load_confusion, score_trace


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/initial_histogram_pool_v1/initial_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/r2_3_multidefect_stratified"
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EPOCHS = 1000


def stratum(n_regions: int) -> str:
    return "1" if n_regions == 1 else "2" if n_regions == 2 else "3-4"


def summarize_group(rows: list[dict]) -> dict:
    gt = sum(r["gt"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    defective = [r for r in rows if r["gt"] > 0]
    return {
        "images": len(rows),
        "mean_complete_image_dice": sum(r["complete_image_dice"] for r in rows) / len(rows),
        "pixel_defect_recall": tp / gt,
        "image_complete_miss_rate": sum(r["complete_miss"] for r in defective) / len(defective),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    confusion = load_confusion(CONFUSION_PATH)
    defect_regions = {
        item["image_id"]: sum(float(x) > 0 for x in item["sub_pixels"])
        for item in pool
    }
    results = []
    for method in ("EDF-Dynamic-Reservation", "HBTASP-Dynamic"):
        for period in PERIODS:
            for lines in LINES:
                for seed in SEEDS:
                    if method == "EDF-Dynamic-Reservation":
                        _, trace = run_edf(pool, period, lines, EPOCHS, seed, "dynamic_reservation")
                    else:
                        _, trace = run_continuous_hbtasp(
                            pool, period, lines, EPOCHS, seed,
                            budget_mode="assigned_level_sensitivity", network_mode="dynamic"
                        )
                    _, image_rows = score_trace(trace, pool, confusion)
                    source_by_uid = {}
                    for event in trace:
                        if "image_uid" in event and "source_image_id" in event:
                            source_by_uid[event["image_uid"]] = event["source_image_id"]
                    groups = defaultdict(list)
                    for row in image_rows:
                        source = source_by_uid[row["image_uid"]]
                        groups[stratum(defect_regions[source])].append(row)
                    for label, rows in groups.items():
                        results.append({
                            "method": method, "period_ms": period, "lines": lines,
                            "seed": seed, "epochs": EPOCHS, "defective_region_stratum": label,
                            **summarize_group(rows),
                        })

    raw_path = OUT / "cell_strata.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary = []
    for method in ("EDF-Dynamic-Reservation", "HBTASP-Dynamic"):
        for label in ("1", "2", "3-4"):
            rows = [r for r in results if r["method"] == method and r["defective_region_stratum"] == label]
            summary.append({
                "method": method,
                "defective_region_stratum": label,
                "cells": len(rows),
                "mean_complete_image_dice": sum(r["mean_complete_image_dice"] for r in rows) / len(rows),
                "pixel_defect_recall": sum(r["pixel_defect_recall"] for r in rows) / len(rows),
                "image_complete_miss_rate": sum(r["image_complete_miss_rate"] for r in rows) / len(rows),
            })
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
