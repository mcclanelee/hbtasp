"""Validate and analyze the paired productive-cooling ablation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
try:
    from experiments.publication_style import apply_publication_style, BLUE, ORANGE
except ImportError:
    from publication_style import apply_publication_style, BLUE, ORANGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v5_productive_cooling_ablation"
KEYS = ["period_ms", "lines", "seed"]
METRICS = ["mandatory_rejection_ratio", "admitted_mandatory_violation_ratio",
           "historical_coverage_adjusted_dice", "mean_complete_image_dice",
           "pixel_defect_recall", "image_complete_miss_rate",
           "peak_modeled_temperature", "mean_voltage"]


def contrast_rows(frame: pd.DataFrame, group_label: str) -> list[dict]:
    """Return seed-clustered paired contrasts for one analysis stratum."""
    seed_means = frame.groupby("seed", as_index=False)[METRICS].mean()
    rows = []
    for metric in METRICS:
        values = seed_means[metric]
        sem = stats.sem(values)
        ci = ((values.mean(), values.mean()) if np.isclose(sem, 0) else
              stats.t.interval(.95, len(values) - 1,
                               loc=values.mean(), scale=sem))
        rows.append({"stratum": group_label, "metric": metric,
                     "cooling_minus_control": values.mean(),
                     "ci95_low": ci[0], "ci95_high": ci[1],
                     "independent_seed_units": len(values)})
    return rows


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    if len(data) != 560 or not (data.groupby("cooling_enabled").size() == 280).all():
        raise RuntimeError("cooling grid is incomplete")
    accounted = (data.completed + data.mandatory_infeasible.fillna(0)
                 + data.optional_skipped.fillna(0) + data.dispatch_infeasible.fillna(0)
                 + data.expired_waiting.fillna(0))
    residual = data.total_regions - accounted
    if np.abs(residual).max() != 0:
        raise RuntimeError("terminal accounting failure")
    if data.thermal_violations.sum() != 0:
        raise RuntimeError("nominal thermal violation")
    off = data[~data.cooling_enabled].set_index(KEYS).sort_index()
    on = data[data.cooling_enabled].set_index(KEYS).sort_index()
    if not off.index.equals(on.index):
        raise RuntimeError("unpaired cooling cells")

    rows = []
    for metric in METRICS:
        delta = on[metric] - off[metric]
        sem = stats.sem(delta)
        ci = ((delta.mean(), delta.mean()) if np.isclose(sem, 0) else
              stats.t.interval(.95, len(delta) - 1, loc=delta.mean(), scale=sem))
        rows.append({"metric": metric, "cooling_minus_control": delta.mean(),
                     "ci95_low": ci[0], "ci95_high": ci[1],
                     "paired_t_p": (1.0 if np.isclose(sem, 0)
                                    else stats.ttest_rel(on[metric], off[metric]).pvalue),
                     "cooling_higher": int((delta > 0).sum()),
                     "ties": int(np.isclose(delta, 0).sum()),
                     "cooling_lower": int((delta < 0).sum())})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(OUT / "paired_contrasts.csv", index=False)
    summary = data.groupby("cooling_enabled")[METRICS].agg(["mean", "std"])
    summary.to_csv(OUT / "configuration_summary.csv")

    paired = on[METRICS].subtract(off[METRICS]).reset_index()
    paired.to_csv(OUT / "paired_cell_differences.csv", index=False)

    # The inferential unit is the seed, not each period-line cell.  Averaging
    # within seed prevents the repeated grid cells from being treated as 280
    # independent replications.
    seed_contrasts = pd.DataFrame(contrast_rows(paired, "all_periods"))
    seed_contrasts.to_csv(OUT / "paired_seed_contrasts.csv", index=False)
    period_rows = []
    for period, group in paired.groupby("period_ms"):
        period_rows.extend(contrast_rows(group, f"period_{int(period)}ms"))
    period_effects = pd.DataFrame(period_rows)
    period_effects.insert(0, "period_ms",
                          period_effects.stratum.str.extract(r"(\d+)")[0].astype(int))
    period_effects.to_csv(OUT / "period_effects.csv", index=False)
    condition_rows = []
    for (period, lines), group in paired.groupby(["period_ms", "lines"]):
        for metric in METRICS:
            values = group[metric]
            sem = stats.sem(values)
            ci = ((values.mean(), values.mean()) if np.isclose(sem, 0) else
                  stats.t.interval(.95, len(values)-1, loc=values.mean(), scale=sem))
            condition_rows.append({"period_ms": period, "lines": lines,
                                   "metric": metric, "cooling_minus_control": values.mean(),
                                   "ci95_low": ci[0], "ci95_high": ci[1]})
    condition_effects = pd.DataFrame(condition_rows)
    condition_effects.to_csv(OUT / "condition_effects.csv", index=False)

    panels = [("mandatory_rejection_ratio", "Mandatory rejection"),
              ("mean_complete_image_dice", r"Complete-image $D_{CI}$"),
              ("peak_modeled_temperature", r"Peak modeled temperature ($^\circ$C)"),
              ("mean_voltage", "Mean selected voltage (V)")]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
    apply_publication_style()
    labels, colors = ["Control", "Low-voltage cooling"], [BLUE, ORANGE]
    for ax, (metric, ylabel) in zip(axes.flat, panels):
        means = [off[metric].mean(), on[metric].mean()]
        errors = [off[metric].std(ddof=1), on[metric].std(ddof=1)]
        bars = ax.bar(np.arange(2), means, yerr=errors, capsize=4, width=.38,
                      color=colors, alpha=.92, edgecolor="#333333", linewidth=.8,
                      error_kw={"elinewidth": 1.3, "capthick": 1.3,
                                "ecolor": "#333333"})
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
        low = min(0, min(m - e for m, e in zip(means, errors)))
        high = max(m + e for m, e in zip(means, errors))
        ax.set_ylim(low, high + max((high - low) * .20, .01))
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (high-low)*.045,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=11,
                    fontweight="semibold")
    fig.savefig(OUT / "r2_1_productive_cooling_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_1_productive_cooling_ablation.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    periods = sorted(paired.period_ms.unique()); line_values = sorted(paired.lines.unique())
    for ax, metric, label, cmap in [
        (axes[0], "peak_modeled_temperature", r"Peak temperature difference ($^\circ$C)", "coolwarm"),
        (axes[1], "mean_voltage", "Mean-voltage difference (V)", "coolwarm"),
    ]:
        matrix = paired.pivot_table(index="lines", columns="period_ms", values=metric,
                                    aggfunc="mean").loc[line_values, periods]
        limit = max(abs(matrix.to_numpy().min()), abs(matrix.to_numpy().max()), 1e-9)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(periods)), periods, rotation=30)
        ax.set_yticks(range(len(line_values)), line_values)
        ax.set_xlabel("Period / deadline (ms)"); ax.set_ylabel("Production lines")
        for i in range(len(line_values)):
            for j in range(len(periods)):
                ax.text(j, i, f"{matrix.iloc[i,j]:+.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label=label)
    fig.savefig(OUT / "r2_1_productive_cooling_load_interaction.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_1_productive_cooling_load_interaction.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    c = seed_contrasts.set_index("metric")
    c200 = period_effects[period_effects.period_ms == 200].set_index("metric")
    report = f"""# Productive-cooling same-kernel ablation

All 560 paired cells are present, terminal accounting is exact, and nominal
thermal violations are zero. The control differs only by disabling the
productive-cooling search branch.

Cooling-minus-control differences are
{c.loc['mandatory_rejection_ratio','cooling_minus_control']:+.6f} for mandatory
rejection, {c.loc['mean_complete_image_dice','cooling_minus_control']:+.6f} for
complete-image Dice,
{c.loc['peak_modeled_temperature','cooling_minus_control']:+.6f} degrees C for
peak modeled temperature, and
{c.loc['mean_voltage','cooling_minus_control']:+.6f} V for mean selected voltage.
These paired effects, including their confidence intervals in
`paired_seed_contrasts.csv`, define the evidential scope of the cooling claim.
The confidence intervals use the ten seeds as independent inferential units;
`paired_contrasts.csv` is retained only as a cell-level sensitivity summary.

At 200 ms, cooling-minus-control differences are
{c200.loc['mandatory_rejection_ratio','cooling_minus_control']:+.6f} for
mandatory rejection,
{c200.loc['admitted_mandatory_violation_ratio','cooling_minus_control']:+.6f}
for admitted mandatory deadline failure,
{c200.loc['peak_modeled_temperature','cooling_minus_control']:+.8f} degrees C
for peak modeled temperature, and
{c200.loc['mean_voltage','cooling_minus_control']:+.6f} V for mean selected
voltage. Thus aggregation across periods does not conceal a service-level or
thermal benefit at the historical 200-ms operating point.
"""
    (OUT / "ANALYSIS.md").write_text(report, encoding="utf-8")
    cp_path = OUT / "checkpoint.json"
    cp = json.loads(cp_path.read_text(encoding="utf-8"))
    cp.update({"status": "complete_validated", "validated_cells": len(data),
               "max_terminal_accounting_residual": float(np.abs(residual).max()),
               "total_thermal_violations": int(data.thermal_violations.sum())})
    cp_path.write_text(json.dumps(cp, indent=2), encoding="utf-8")
    print(summary.to_string()); print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
