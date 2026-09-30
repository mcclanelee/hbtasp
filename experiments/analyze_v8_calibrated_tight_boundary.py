"""Validate and summarize the calibrated V8 tight-period boundary.

Confidence intervals use the same seed-clustered estimand as the main V8
factorial: paired HBTASP-minus-EDF differences are first averaged over all
period/line cells within each seed, and the ten seed means are treated as the
independent replication units.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
try:
    from experiments.publication_style import apply_publication_style
except ImportError:
    from publication_style import apply_publication_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v8_calibrated_tight_boundary"
KEYS = ["period_ms", "lines", "seed"]
HBT = "HBTASP-Dynamic"
EDF = "EDF-Dynamic-Reservation"


def clustered_summary(delta: pd.Series) -> dict[str, float | int]:
    by_seed = delta.groupby(level="seed").mean()
    sem = stats.sem(by_seed)
    ci = stats.t.interval(0.95, len(by_seed) - 1, loc=by_seed.mean(), scale=sem)
    return {
        "hbt_minus_edf": float(by_seed.mean()),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "paired_seed_t_p": float(stats.ttest_1samp(by_seed, 0).pvalue),
        "n_seed_clusters": int(len(by_seed)),
        "hbt_higher_cells": int((delta > 0).sum()),
        "ties_cells": int(np.isclose(delta, 0).sum()),
        "hbt_lower_cells": int((delta < 0).sum()),
    }


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    if len(data) != 700 or not (data.groupby("configuration").size() == 350).all():
        raise RuntimeError("incomplete tight-period grid")

    is_edf = data.configuration == EDF
    edf_account = data.completed + data.expired_waiting + data.admission_infeasible.fillna(0)
    hbt_account = (
        data.completed
        + data.expired_waiting
        + data.mandatory_infeasible.fillna(0)
        + data.optional_skipped.fillna(0)
        + data.dispatch_infeasible.fillna(0)
    )
    residual = np.where(is_edf, data.total_regions - edf_account, data.total_regions - hbt_account)
    if np.abs(residual).max() != 0:
        raise RuntimeError("terminal accounting failure")
    if data.thermal_violations.sum() != 0:
        raise RuntimeError("nominal thermal violation")

    metrics = [
        "mandatory_dmr",
        "historical_coverage_adjusted_dice",
        "historical_weighted_coverage_utility",
        "historical_mandatory_effective_dice",
        "mean_complete_image_dice",
        "pixel_defect_recall",
        "image_complete_miss_rate",
    ]
    means = data.groupby(["period_ms", "lines", "configuration"])[metrics].mean().reset_index()
    means.to_csv(OUT / "cell_means.csv", index=False)

    wide = data.pivot(index=KEYS, columns="configuration", values=metrics)
    paired = []
    for metric in metrics:
        delta = wide[metric][HBT] - wide[metric][EDF]
        row = {"metric": metric, **clustered_summary(delta)}
        paired.append(row)
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(OUT / "paired_overall_seed_clustered.csv", index=False)

    grid = means.pivot(index=["period_ms", "lines"], columns="configuration", values=metrics)
    boundary = pd.DataFrame(index=grid.index).reset_index()
    for metric in metrics:
        boundary[f"delta_{metric}"] = (grid[metric][HBT] - grid[metric][EDF]).to_numpy()
    boundary.to_csv(OUT / "boundary_differences.csv", index=False)

    by_period = data.groupby(["period_ms", "configuration"])[metrics].mean().reset_index()
    by_period.to_csv(OUT / "period_means.csv", index=False)

    apply_publication_style()
    panels = [
        ("delta_mandatory_dmr", "Mandatory-region service failure\n(HBTASP - EDF)"),
        ("delta_historical_weighted_coverage_utility", "Weighted utility\n(HBTASP - EDF)"),
        ("delta_mean_complete_image_dice", r"$D_{CI}$\n(HBTASP - EDF)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    for ax, (column, label) in zip(axes, panels):
        matrix = boundary.pivot(index="lines", columns="period_ms", values=column).sort_index(ascending=False)
        vmax = np.abs(matrix.to_numpy()).max()
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("Relative deadline / period (ms)")
        ax.set_ylabel("Production lines")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > 0.55 * vmax else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label)
    fig.savefig(OUT / "r2_7_calibrated_tight_period_boundary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_7_calibrated_tight_period_boundary.png", bbox_inches="tight")
    plt.close(fig)

    p = paired_df.set_index("metric")
    report = f"""# V8 calibrated tight-period dynamic boundary

All 700 cells are present, terminal-accounting residual is zero, and nominal
thermal violations are zero. Across 50--90 ms and 4--10 lines, HBTASP minus
EDF has mandatory-region service-failure difference {p.loc['mandatory_dmr','hbt_minus_edf']:.5f}
(seed-clustered 95% CI {p.loc['mandatory_dmr','ci95_low']:.5f} to
{p.loc['mandatory_dmr','ci95_high']:.5f}), complete-image Dice difference
{p.loc['mean_complete_image_dice','hbt_minus_edf']:.5f} (95% CI
{p.loc['mean_complete_image_dice','ci95_low']:.5f} to
{p.loc['mean_complete_image_dice','ci95_high']:.5f}), recall difference
{p.loc['pixel_defect_recall','hbt_minus_edf']:.5f} (95% CI
{p.loc['pixel_defect_recall','ci95_low']:.5f} to
{p.loc['pixel_defect_recall','ci95_high']:.5f}), and image-miss difference
{p.loc['image_complete_miss_rate','hbt_minus_edf']:.5f} (95% CI
{p.loc['image_complete_miss_rate','ci95_low']:.5f} to
{p.loc['image_complete_miss_rate','ci95_high']:.5f}).

The grid is an operating-boundary result, not universal-dominance evidence.
Negative mandatory-region service-failure and image-miss differences favor HBTASP; positive Dice
and recall differences favor HBTASP. Intervals use ten paired seed clusters.
"""
    (OUT / "ANALYSIS.md").write_text(report, encoding="utf-8")

    cp = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    cp.update({
        "status": "complete_validated",
        "validated_cells": len(data),
        "max_terminal_accounting_residual": float(np.abs(residual).max()),
        "total_thermal_violations": int(data.thermal_violations.sum()),
        "inference_unit": "10 paired seed clusters",
    })
    (OUT / "checkpoint.json").write_text(json.dumps(cp, indent=2), encoding="utf-8")
    print(paired_df.to_string(index=False))
    print("\nBy period:\n", by_period.to_string(index=False))


if __name__ == "__main__":
    main()
