"""Seed-clustered contrasts for calibrated-priority multi-defect replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v8_calibrated_multidefect_stratified"
METRICS = ("mean_complete_image_dice", "pixel_defect_recall", "image_complete_miss_rate")


def main():
    data = pd.read_csv(OUT / "cell_strata.csv")
    if len(data) != 1200 or not (data.groupby(["method", "defective_region_stratum"]).size() == 200).all():
        raise RuntimeError("incomplete stratified grid")
    result = {}
    for label in ("1", "2", "3-4"):
        result[label] = {}
        part = data[data.defective_region_stratum.astype(str) == label]
        seed = part.groupby(["seed", "method"])[list(METRICS)].mean().reset_index()
        for metric in METRICS:
            wide = seed.pivot(index="seed", columns="method", values=metric)
            delta = wide["HBTASP-Dynamic"] - wide["EDF-Dynamic-Reservation"]
            ci = stats.t.interval(.95, len(delta)-1, loc=delta.mean(), scale=stats.sem(delta))
            result[label][metric] = {"mean_difference": float(delta.mean()),
                                     "ci95": [float(ci[0]), float(ci[1])],
                                     "n_seeds": len(delta)}
    (OUT / "clustered_contrasts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
