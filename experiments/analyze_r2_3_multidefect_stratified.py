"""Seed-clustered paired contrasts for the R2.3 stratified diagnostic."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "experiments/checkpoints/r2_3_multidefect_stratified/cell_strata.csv"
OUTFILE = ROOT / "experiments/checkpoints/r2_3_multidefect_stratified/clustered_contrasts.json"
METRICS = ("mean_complete_image_dice", "pixel_defect_recall", "image_complete_miss_rate")


with INFILE.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

cluster = defaultdict(list)
for row in rows:
    cluster[(row["seed"], row["method"], row["defective_region_stratum"])].append(row)

result = {}
for stratum in ("1", "2", "3-4"):
    result[stratum] = {}
    for metric in METRICS:
        differences = []
        for seed in sorted({r["seed"] for r in rows}, key=int):
            def mean(method):
                values = cluster[(seed, method, stratum)]
                return sum(float(x[metric]) for x in values) / len(values)
            differences.append(mean("HBTASP-Dynamic") - mean("EDF-Dynamic-Reservation"))
        n = len(differences)
        average = sum(differences) / n
        sd = math.sqrt(sum((x - average) ** 2 for x in differences) / (n - 1))
        half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        result[stratum][metric] = {
            "contrast": "HBTASP-Dynamic - EDF-Dynamic-Reservation",
            "mean_difference": average,
            "ci95": [average - half, average + half],
            "n_seeds": n,
        }

OUTFILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
