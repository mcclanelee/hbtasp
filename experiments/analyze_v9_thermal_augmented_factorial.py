"""Validate the ATP/IIT augmentation and summarize it for the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v9_thermal_augmented_factorial"
AUTH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/cell_results.csv"
KEYS = ["configuration", "period_ms", "lines", "seed"]


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    auth = pd.read_csv(AUTH)
    if len(data) != 800 or data[KEYS].duplicated().any():
        raise RuntimeError("incomplete or duplicated V9 grid")
    merged = data.merge(auth[KEYS + ["mandatory_dmr", "mean_complete_image_dice"]],
                        on=KEYS, suffixes=("_v9", "_v8"), validate="one_to_one")
    dmr_error = np.abs(merged.mandatory_dmr_v9 - merged.mandatory_dmr_v8).max()
    dice_error = np.abs(merged.mean_complete_image_dice_v9 - merged.mean_complete_image_dice_v8).max()
    if dmr_error > 1e-12 or dice_error > 1e-12:
        raise RuntimeError("V9 changed an authoritative scheduling metric")
    if data.thermal_violations.sum() != 0 or data.iit_celsius_seconds.max() > 1e-12:
        raise RuntimeError("nominal thermal invariant failed")
    if data.peak_temperature_c.max() > 60.0 + 1e-8:
        raise RuntimeError("reconstructed nominal peak exceeds Tmax")

    metrics = ["mandatory_dmr", "mean_complete_image_dice", "average_temperature_c",
               "iit_celsius_seconds", "peak_temperature_c"]
    period = data.groupby(["configuration", "period_ms"])[metrics].mean().reset_index()
    overall = data.groupby("configuration")[metrics].mean().reset_index()
    period.to_csv(OUT / "period_summary.csv", index=False)
    overall.to_csv(OUT / "overall_summary.csv", index=False)

    report = [
        "# V9 thermal-augmented calibrated factorial", "",
        "All 800 cells are present. Mandatory DMR and complete-image Dice are",
        "identical cell by cell to the authoritative V8 factorial. The continuous",
        "RC reconstruction has zero nominal IIT, zero thermal violations, and no",
        "peak above 60 C. ATP is therefore an additional descriptive trajectory",
        "metric, not a rerun with altered scheduling decisions.", "", "## Overall", "",
        "```", overall.to_string(index=False), "```", "", "## By period", "",
        "```", period.to_string(index=False), "```",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(report), encoding="utf-8")
    cp = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    cp.update({
        "status": "complete_validated", "validated_cells": 800,
        "max_v8_dmr_difference": float(dmr_error),
        "max_v8_dci_difference": float(dice_error),
        "total_thermal_violations": int(data.thermal_violations.sum()),
        "max_iit_celsius_seconds": float(data.iit_celsius_seconds.max()),
        "max_peak_temperature_c": float(data.peak_temperature_c.max()),
    })
    (OUT / "checkpoint.json").write_text(json.dumps(cp, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    print("\n", period.to_string(index=False))


if __name__ == "__main__":
    main()
