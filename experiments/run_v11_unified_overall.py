"""Run the frozen submitted-family Overall V11 protocol."""

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
from experiments.initial_r2_8_metrics import (
    load_confusion, score_historical_scalar, score_trace,
)
from experiments.overall_baseline_adapters import (
    run_esatd_l3, run_heat_l3, run_static_v,
)
from experiments.run_v9_thermal_augmented_factorial import reconstruct_thermal


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/OVERALL_PROTOCOL_V11.json"
POOL_PATH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v11_unified_overall"
RESULT = OUT / "cell_results.csv"
CONFIGS = ("Static-V", "ESATD-L3", "HEAT-L3", "Full-HBTASP")
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


def run_key(key: tuple[str, int, int, int]) -> dict:
    config, period, lines, seed = key
    if config == "Static-V":
        summary, trace = run_static_v(_POOL, period, lines, EPOCHS, seed)
    elif config == "ESATD-L3":
        summary, trace = run_esatd_l3(_POOL, period, lines, EPOCHS, seed)
    elif config == "HEAT-L3":
        summary, trace = run_heat_l3(_POOL, period, lines, EPOCHS, seed)
    elif config == "Full-HBTASP":
        summary, trace = run_continuous_hbtasp(
            _POOL, period, lines, EPOCHS, seed,
            budget_mode="assigned_level_sensitivity", network_mode="dynamic",
            enable_productive_cooling=True,
        )
    else:
        raise ValueError(config)

    if summary["mandatory_terminal_residual"] != 0:
        raise RuntimeError(f"terminal residual for {key}")
    if summary["mandatory_unique_terminal_uids"] != summary["mandatory_released"]:
        raise RuntimeError(f"terminal UID mismatch for {key}")
    scalar = score_historical_scalar(trace, summary["total_regions"])
    mask, _ = score_trace(trace, _POOL, _CONFUSION)
    thermal = reconstruct_thermal(trace)
    return {
        "configuration": config, "period_ms": period, "lines": lines,
        "seed": seed, "epochs": EPOCHS,
        **summary, **scalar, **mask, **thermal,
    }


def write_rows(rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = RESULT.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(RESULT)


def checkpoint(rows: list[dict], started: float) -> None:
    payload = {
        "status": "complete" if len(rows) == len(CONFIGS)*len(PERIODS)*len(LINES)*len(SEEDS) else "running",
        "completed_cells": len(rows),
        "total_cells": len(CONFIGS)*len(PERIODS)*len(LINES)*len(SEEDS),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_this_run": time.time() - started,
        "protocol_id": "hbtasp_overall_v11",
        "protocol_sha256": sha256(PROTOCOL),
        "pool_sha256": sha256(POOL_PATH),
        "confusion_sha256": sha256(CONFUSION_PATH),
        "adapter_sha256": sha256(ROOT / "experiments/overall_baseline_adapters.py"),
        "terminal_registry_sha256": sha256(ROOT / "experiments/mandatory_terminal_registry.py"),
        "result_sha256": sha256(RESULT),
    }
    (OUT / "checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RESULT).to_dict("records") if RESULT.exists() else []
    done = {(r["configuration"], int(r["period_ms"]), int(r["lines"]), int(r["seed"])) for r in rows}
    keys = [(c,p,l,s) for c in CONFIGS for p in PERIODS for l in LINES for s in SEEDS]
    pending = [key for key in keys if key not in done]
    started = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = {executor.submit(run_key, key): key for key in pending}
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda r: (CONFIGS.index(r["configuration"]), int(r["period_ms"]),
                                     int(r["lines"]), int(r["seed"])))
            write_rows(rows); checkpoint(rows, started)
            if len(rows) % 20 == 0 or len(rows) == len(keys):
                print(f"[{len(rows)}/{len(keys)}]", flush=True)


if __name__ == "__main__":
    main()
