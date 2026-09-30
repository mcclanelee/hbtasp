"""Validate, summarize, and plot the V5 tight-period dynamic boundary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v5_tight_period_dynamic_boundary"
KEYS = ["period_ms", "lines", "seed"]
HBT = "HBTASP-Dynamic"; EDF = "EDF-Dynamic-Reservation"


def main():
    data = pd.read_csv(OUT / "cell_results.csv")
    if len(data) != 700 or not (data.groupby("configuration").size() == 350).all():
        raise RuntimeError("incomplete tight-period grid")
    edf = data.configuration == EDF
    edf_account = data.completed + data.expired_waiting + data.admission_infeasible.fillna(0)
    hbt_account = (data.completed + data.expired_waiting + data.mandatory_infeasible.fillna(0)
                   + data.optional_skipped.fillna(0) + data.dispatch_infeasible.fillna(0))
    residual = np.where(edf, data.total_regions - edf_account, data.total_regions - hbt_account)
    if np.abs(residual).max() != 0: raise RuntimeError("terminal accounting failure")
    if data.thermal_violations.sum() != 0: raise RuntimeError("nominal thermal violation")

    metrics = ["mandatory_dmr", "historical_coverage_adjusted_dice",
               "historical_weighted_coverage_utility", "historical_mandatory_effective_dice",
               "mean_complete_image_dice", "pixel_defect_recall", "image_complete_miss_rate"]
    means = data.groupby(["period_ms", "lines", "configuration"])[metrics].mean().reset_index()
    means.to_csv(OUT / "cell_means.csv", index=False)
    wide = data.pivot(index=KEYS, columns="configuration", values=metrics)
    paired = []
    for metric in metrics:
        delta = wide[metric][HBT] - wide[metric][EDF]
        sem = stats.sem(delta); ci = stats.t.interval(.95, len(delta)-1, loc=delta.mean(), scale=sem)
        paired.append({"metric": metric, "hbt_minus_edf": delta.mean(), "ci95_low": ci[0],
                       "ci95_high": ci[1], "paired_t_p": stats.ttest_rel(wide[metric][HBT], wide[metric][EDF]).pvalue,
                       "hbt_higher": int((delta > 0).sum()), "ties": int(np.isclose(delta, 0).sum()),
                       "hbt_lower": int((delta < 0).sum())})
    pd.DataFrame(paired).to_csv(OUT / "paired_overall.csv", index=False)

    grid = means.pivot(index=["period_ms", "lines"], columns="configuration", values=metrics)
    boundary = pd.DataFrame(index=grid.index).reset_index()
    for metric in metrics:
        boundary[f"delta_{metric}"] = (grid[metric][HBT] - grid[metric][EDF]).to_numpy()
    boundary.to_csv(OUT / "boundary_differences.csv", index=False)

    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica"],
        "font.size":11, "axes.labelsize":12, "xtick.labelsize":10, "ytick.labelsize":10,
        "figure.dpi":300, "savefig.dpi":300, "pdf.fonttype":42})
    panels = [("delta_mandatory_dmr", "Mandatory DMR\n(HBTASP - EDF)"),
              ("delta_historical_weighted_coverage_utility", "Weighted utility\n(HBTASP - EDF)"),
              ("delta_mean_complete_image_dice", r"$D_{CI}$\n(HBTASP - EDF)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    for ax, (column, label) in zip(axes, panels):
        matrix = boundary.pivot(index="lines", columns="period_ms", values=column).sort_index(ascending=False)
        vmax = np.abs(matrix.to_numpy()).max()
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("Relative deadline / period (ms)"); ax.set_ylabel("Production lines")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > 0.55*vmax else "black")
        cb = fig.colorbar(image, ax=ax, fraction=.046, pad=.03); cb.set_label(label)
    fig.savefig(OUT / "r2_7_tight_period_boundary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_7_tight_period_boundary.png", bbox_inches="tight")
    plt.close(fig)

    p = pd.DataFrame(paired).set_index("metric")
    report = f"""# V5 tight-period dynamic boundary

All 700 cells are present, terminal-accounting residual is zero, and nominal
thermal violations are zero. Across 50--90 ms and 4--10 lines, HBTASP minus
EDF has mandatory-DMR difference {p.loc['mandatory_dmr','hbt_minus_edf']:.5f}
(95% CI {p.loc['mandatory_dmr','ci95_low']:.5f} to {p.loc['mandatory_dmr','ci95_high']:.5f}),
weighted-utility difference {p.loc['historical_weighted_coverage_utility','hbt_minus_edf']:.5f}
(95% CI {p.loc['historical_weighted_coverage_utility','ci95_low']:.5f} to
{p.loc['historical_weighted_coverage_utility','ci95_high']:.5f}), and complete-image
Dice difference {p.loc['mean_complete_image_dice','hbt_minus_edf']:.5f}.

The grid is a policy-boundary result, not universal dominance evidence. Negative
mandatory-DMR differences favor HBTASP; positive quality differences favor HBTASP.
"""
    (OUT / "ANALYSIS.md").write_text(report, encoding="utf-8")
    cp = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
    cp.update({"status":"complete_validated", "validated_cells":len(data),
               "max_terminal_accounting_residual":float(np.abs(residual).max()),
               "total_thermal_violations":int(data.thermal_violations.sum())})
    (OUT / "checkpoint.json").write_text(json.dumps(cp, indent=2), encoding="utf-8")
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__": main()
