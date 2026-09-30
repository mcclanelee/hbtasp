"""Top-K mandatory-region sweep on the frozen calibrated HBTASP protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import load_confusion, score_trace


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v19_topk_mandatory_sweep"
RESULT = OUT / "cell_results.csv"
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
MANDATORY_COUNTS = (1, 2, 3, 4)
EPOCHS = 1000

_POOL = None
_CONFUSION = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init_worker() -> None:
    global _POOL, _CONFUSION
    _POOL = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    _CONFUSION = load_confusion(CONFUSION_PATH)


def coverage_metrics(trace: list[dict], pool: list[dict]) -> dict[str, float]:
    """Separate scheduling coverage of defect-positive and optional regions."""
    pool_by_id = {str(item["image_id"]): item for item in pool}
    terminal_events = {
        "complete", "mandatory_infeasible", "optional_skipped", "expired",
        "dispatch_infeasible",
    }
    by_image: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    optional_total = optional_on_time = 0
    defect_total = defect_on_time = defect_mandatory = 0
    seen = set()
    for event in trace:
        if event.get("event") not in terminal_events:
            continue
        uid = event["uid"]
        if uid in seen:
            raise RuntimeError(f"duplicate terminal event: {uid}")
        seen.add(uid)
        source = str(event["source_image_id"])
        region = int(event["region_index"])
        positive = int(pool_by_id[source]["sub_pixels"][region]) > 0
        mandatory = bool(event.get("mandatory"))
        on_time = event["event"] == "complete" and not event.get("deadline_miss", False)
        by_image[event["image_uid"]].append((positive, mandatory, on_time))
        if not mandatory:
            optional_total += 1
            optional_on_time += int(on_time)
        if positive:
            defect_total += 1
            defect_on_time += int(on_time)
            defect_mandatory += int(mandatory)

    defective_images = 0
    all_defect_on_time = 0
    any_defect_on_time = 0
    for records in by_image.values():
        positives = [record for record in records if record[0]]
        if not positives:
            continue
        defective_images += 1
        all_defect_on_time += int(all(record[2] for record in positives))
        any_defect_on_time += int(any(record[2] for record in positives))

    return {
        "optional_region_on_time_coverage": optional_on_time / optional_total if optional_total else float("nan"),
        "defect_region_mandatory_coverage": defect_mandatory / defect_total,
        "defect_region_on_time_coverage": defect_on_time / defect_total,
        "all_defect_regions_on_time_image_rate": all_defect_on_time / defective_images,
        "any_defect_region_on_time_image_rate": any_defect_on_time / defective_images,
        "defect_positive_regions": defect_total,
        "defective_images_for_coverage": defective_images,
    }


def run_key(key: tuple[int, int, int, int]) -> dict:
    mandatory_count, period, lines, seed = key
    summary, trace = run_continuous_hbtasp(
        _POOL, period, lines, EPOCHS, seed,
        budget_mode="assigned_level_sensitivity",
        network_mode="dynamic",
        mandatory_count=mandatory_count,
    )
    mask, _ = score_trace(trace, _POOL, _CONFUSION)
    coverage = coverage_metrics(trace, _POOL)
    return {
        "configuration": f"HBTASP-Top{mandatory_count}",
        "mandatory_count": mandatory_count,
        **summary,
        **mask,
        **coverage,
    }


def write_rows(rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = RESULT.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(RESULT)


def write_checkpoint(rows: list[dict], total: int, elapsed: float) -> None:
    payload = {
        "status": "complete" if len(rows) == total else "running",
        "completed_cells": len(rows),
        "total_cells": total,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_this_run": elapsed,
        "protocol": {
            "mandatory_counts": MANDATORY_COUNTS,
            "periods_ms": PERIODS,
            "lines": LINES,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "priority_pool": "frozen_v8_calibrated_priority_pool",
            "network_mode": "dynamic_l1_l5",
        },
        "sha256": {
            "pool": sha256(POOL_PATH),
            "confusion": sha256(CONFUSION_PATH),
            "event_kernel": sha256(ROOT / "experiments/initial_manuscript_event_replay.py"),
            "runner": sha256(Path(__file__)),
            "results": sha256(RESULT),
        },
    }
    (OUT / "checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RESULT).to_dict("records") if RESULT.exists() else []
    done = {
        (int(row["mandatory_count"]), int(row["period_ms"]), int(row["lines"]), int(row["seed"]))
        for row in rows
    }
    keys = [
        (count, period, lines, seed)
        for count in MANDATORY_COUNTS
        for period in PERIODS
        for lines in LINES
        for seed in SEEDS
    ]
    started = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = {executor.submit(run_key, key): key for key in keys if key not in done}
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda row: (
                int(row["mandatory_count"]), int(row["period_ms"]),
                int(row["lines"]), int(row["seed"]),
            ))
            write_rows(rows)
            write_checkpoint(rows, len(keys), time.time() - started)
            if len(rows) % 20 == 0:
                print(f"[{len(rows)}/{len(keys)}]", flush=True)
    write_checkpoint(rows, len(keys), time.time() - started)


if __name__ == "__main__":
    main()
