"""Matched all-regions-mandatory HBTASP boundary for the full-image audit."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import load_confusion, score_trace
from experiments.run_v5_strict_all_regions_boundary import strict_stats

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v14_strict_all_regions_matched"
RESULT = OUT / "cell_results.csv"
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EPOCHS = 1000
_POOL = None
_CONFUSION = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init_worker() -> None:
    global _POOL, _CONFUSION
    _POOL = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    _CONFUSION = load_confusion(CONFUSION_PATH)


def run_key(key: tuple[int, int, int]) -> dict:
    period, lines, seed = key
    summary, trace = run_continuous_hbtasp(
        _POOL, period, lines, EPOCHS, seed,
        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
        all_regions_mandatory=True,
    )
    mask, _ = score_trace(trace, _POOL, _CONFUSION)
    row = {
        **summary, **strict_stats(trace, lines * EPOCHS), **mask,
        "evidence_pool": "frozen_v8_calibrated_priority_pool",
    }
    residual = (row["total_regions"] - row["pre_execution_rejections"]
                - row["admitted_deadline_violations"]
                - (row["completed"] - row["deadline_misses"]))
    if residual != 0:
        raise RuntimeError(f"terminal residual {residual}: {key}")
    row["terminal_residual"] = residual
    return row


def write_rows(rows: list[dict]) -> None:
    fields = []
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


def checkpoint(rows: list[dict], total: int, elapsed: float) -> None:
    payload = {
        "status": "complete" if len(rows) == total else "running",
        "completed_cells": len(rows), "total_cells": total,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_this_run": elapsed,
        "protocol": {"periods_ms": PERIODS, "lines": LINES, "seeds": SEEDS,
                     "epochs": EPOCHS, "all_regions_mandatory": True,
                     "release": "periodic", "priority_overhead_ms": 0.0},
        "sha256": {"pool": sha256(POOL_PATH), "confusion": sha256(CONFUSION_PATH),
                   "event_kernel": sha256(ROOT / "experiments/initial_manuscript_event_replay.py"),
                   "hbtasp": sha256(ROOT / "experiments/initial_manuscript_hbtasp.py"),
                   "runner": sha256(Path(__file__)), "results": sha256(RESULT)},
    }
    (OUT / "checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RESULT).to_dict("records") if RESULT.exists() else []
    done = {(int(r["period_ms"]), int(r["lines"]), int(r["seed"])) for r in rows}
    keys = [(p, n, s) for p in PERIODS for n in LINES for s in SEEDS]
    started = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = {executor.submit(run_key, key): key for key in keys if key not in done}
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda r: (int(r["period_ms"]), int(r["lines"]), int(r["seed"])))
            write_rows(rows)
            checkpoint(rows, len(keys), time.time() - started)
            if len(rows) % 20 == 0:
                print(f"[{len(rows)}/{len(keys)}]", flush=True)
    if rows:
        checkpoint(rows, len(keys), time.time() - started)


if __name__ == "__main__":
    main()
