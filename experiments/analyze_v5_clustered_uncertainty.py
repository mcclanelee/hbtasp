"""Seed-clustered paired uncertainty for the main scheduler comparison.

The independent inferential unit is the random seed.  Fixed design cells
(period and production-line count) are averaged within each seed before the
paired HBTASP-minus-EDF contrast and Student-t confidence interval are formed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/checkpoints/v5_final_factorial_selected_budget/cell_results.csv"
OUT = SOURCE.parent / "clustered_uncertainty.json"
HBT = "HBTASP-Dynamic"
EDF = "EDF-Dynamic-Reservation"
METRICS = (
    "mandatory_dmr",
    "mean_complete_image_dice",
    "pixel_defect_recall",
    "image_complete_miss_rate",
    "historical_coverage_adjusted_dice",
)


def main() -> None:
    data = pd.read_csv(SOURCE)
    selected = data[data["configuration"].isin((EDF, HBT))].copy()
    required = {"configuration", "period_ms", "lines", "seed", *METRICS}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    cell_counts = selected.groupby("configuration").size()
    if len(cell_counts) != 2 or cell_counts.nunique() != 1:
        raise RuntimeError(f"unbalanced scheduler arms: {cell_counts.to_dict()}")

    seed_means = selected.groupby(["configuration", "seed"])[list(METRICS)].mean()
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "cluster": "seed",
        "within_cluster_average": ["period_ms", "lines"],
        "n_clusters": int(selected["seed"].nunique()),
        "contrast": f"{HBT} minus {EDF}",
        "metrics": {},
    }

    for metric in METRICS:
        edf = seed_means.loc[EDF, metric].sort_index()
        hbt = seed_means.loc[HBT, metric].sort_index()
        if not edf.index.equals(hbt.index):
            raise RuntimeError(f"unpaired seeds for {metric}")
        delta = hbt - edf
        sem = stats.sem(delta)
        ci = stats.t.interval(
            0.95, len(delta) - 1, loc=delta.mean(), scale=sem
        )
        result["metrics"][metric] = {
            "edf_mean": float(edf.mean()),
            "hbtasp_mean": float(hbt.mean()),
            "paired_mean_difference": float(delta.mean()),
            "clustered_t_ci95_low": float(ci[0]),
            "clustered_t_ci95_high": float(ci[1]),
            "paired_t_p": float(stats.ttest_1samp(delta, 0).pvalue),
            "cluster_differences": {
                str(int(seed)): float(value) for seed, value in delta.items()
            },
        }

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote seed-clustered audit for {result['n_clusters']} paired seeds to {OUT}")


if __name__ == "__main__":
    main()
