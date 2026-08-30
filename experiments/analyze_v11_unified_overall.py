"""Integrity audit and descriptive summaries for unified Overall V11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "checkpoints" / "v11_unified_overall"
CELLS = OUT / "cell_results.csv"
PROTOCOL = ROOT / "OVERALL_PROTOCOL_V11.json"
CONFIGS = ("Static-V", "ESATD-L3", "HEAT-L3", "Full-HBTASP")
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
KEYS = ["configuration", "period_ms", "lines", "seed"]
METRICS = [
    "mandatory_service_failure_rate", "mandatory_pre_rejection_rate",
    "mandatory_admitted_dmr", "mean_complete_image_dice",
    "pixel_defect_recall", "image_complete_miss_rate",
    "average_temperature_c", "iit_celsius_seconds", "peak_temperature_c",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frame = pd.read_csv(CELLS)
    expected = {
        (config, period, lines, seed)
        for config in CONFIGS for period in PERIODS
        for lines in LINES for seed in SEEDS
    }
    observed = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))

    terminal_sum = frame[[
        "mandatory_pre_reject", "mandatory_wait_expire",
        "mandatory_late_complete", "mandatory_on_time",
    ]].sum(axis=1)
    finite_metrics = np.isfinite(frame[METRICS].to_numpy(dtype=float)).all()
    checks = {
        "row_count": int(len(frame)),
        "expected_row_count": len(expected),
        "duplicate_key_count": int(frame.duplicated(KEYS).sum()),
        "missing_keys": [list(item) for item in sorted(expected - observed)],
        "unexpected_keys": [list(item) for item in sorted(observed - expected)],
        "terminal_residual_max_abs": int(frame["mandatory_terminal_residual"].abs().max()),
        "terminal_sum_mismatch_cells": int((terminal_sum != frame["mandatory_released"]).sum()),
        "terminal_uid_mismatch_cells": int((
            frame["mandatory_unique_terminal_uids"] != frame["mandatory_released"]
        ).sum()),
        "nonfinite_metric_cells": int(not finite_metrics),
        "cell_results_sha256": sha256(CELLS),
        "protocol_sha256": sha256(PROTOCOL),
    }
    checks["passed"] = (
        checks["row_count"] == checks["expected_row_count"]
        and checks["duplicate_key_count"] == 0
        and not checks["missing_keys"] and not checks["unexpected_keys"]
        and checks["terminal_residual_max_abs"] == 0
        and checks["terminal_sum_mismatch_cells"] == 0
        and checks["terminal_uid_mismatch_cells"] == 0
        and checks["nonfinite_metric_cells"] == 0
    )

    overall = frame.groupby("configuration", sort=False)[METRICS].agg(["mean", "std"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    overall.reset_index().to_csv(OUT / "overall_summary.csv", index=False)

    period = frame.groupby(["configuration", "period_ms"], sort=False)[METRICS].mean()
    period.reset_index().to_csv(OUT / "period_summary.csv", index=False)

    lines = frame.groupby(["configuration", "lines"], sort=False)[METRICS].mean()
    lines.reset_index().to_csv(OUT / "line_summary.csv", index=False)

    (OUT / "integrity_audit.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))
    if not checks["passed"]:
        raise SystemExit("V11 integrity audit failed")


if __name__ == "__main__":
    main()
