"""Analyze and plot the V10 semantic replay without changing scheduler data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.publication_style import apply_publication_style


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "experiments/checkpoints/v10_mandatory_outcome_audit/cell_results.csv"
OUT = IN.parent
CONFIGS = ["EDF-FixedL3", "EDF-Dynamic-Reservation", "HBTASP-FixedL3", "HBTASP-Dynamic"]
LABELS = ["EDF--Fixed L3", "EDF--Dyn.-Res.", "HBTASP (Fixed L3)", "HBTASP"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]


def weighted_rate(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    return frame[numerator].sum() / frame[denominator].sum()


def main() -> None:
    data = pd.read_csv(IN)
    if len(data) != 800:
        raise RuntimeError(f"expected 800 cells, found {len(data)}")
    if not np.allclose(data.mandatory_dmr, data.mandatory_service_failure_rate):
        raise RuntimeError("legacy label and service-failure metric differ")

    outcome_cols = ["mandatory_completed_on_time", "mandatory_completed_late",
                    "mandatory_expired", "mandatory_dispatch_infeasible",
                    "mandatory_allocation_rejected"]
    summary_rows = []
    for config, group in data.groupby("configuration", sort=False):
        total = group.mandatory_released.sum()
        row = {"configuration": config, "mandatory_released": int(total)}
        for col in outcome_cols:
            row[col] = int(group[col].sum())
            row[col + "_rate"] = (group[col] / group.mandatory_released).mean()
            row[col + "_weighted_rate"] = group[col].sum() / total
        row["mandatory_service_failure_rate"] = group.mandatory_service_failures.sum() / total
        row["cell_mean_mandatory_service_failure_rate"] = group.mandatory_service_failure_rate.mean()
        row["cell_mean_mandatory_on_time_service_rate"] = group.mandatory_on_time_service_rate.mean()
        row["mandatory_completed_job_dmr"] = (
            group.mandatory_completed_late.sum() /
            (group.mandatory_completed_on_time.sum() + group.mandatory_completed_late.sum())
        )
        row["mandatory_admitted_failure_rate"] = (
            (group.mandatory_expired.sum() + group.mandatory_completed_late.sum()) /
            (total - group.mandatory_allocation_rejected.sum() -
             group.mandatory_dispatch_infeasible.sum())
        )
        row["mean_complete_image_dice"] = group.mean_complete_image_dice.mean()
        row["pixel_defect_recall"] = group.pixel_defect_recall.mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).set_index("configuration").loc[CONFIGS].reset_index()
    summary.to_csv(OUT / "mandatory_outcome_summary.csv", index=False)

    # Per-period and per-line feasible region.  A cell is feasible at epsilon if
    # its paired-seed aggregate service failure is within epsilon.
    cell = data.groupby(["configuration", "period_ms", "lines"], as_index=False).agg(
        mandatory_service_failures=("mandatory_service_failures", "sum"),
        mandatory_released=("mandatory_released", "sum"),
        mean_complete_image_dice=("mean_complete_image_dice", "mean"),
        pixel_defect_recall=("pixel_defect_recall", "mean"),
        completed_job_dmr=("mandatory_completed_job_dmr", "mean"),
    )
    cell["mandatory_service_failure_rate"] = (
        cell.mandatory_service_failures / cell.mandatory_released)
    cell.to_csv(OUT / "period_line_service_grid.csv", index=False)
    feasible_rows = []
    for epsilon in (0.0, 0.01, 0.05):
        for config, group in cell.groupby("configuration"):
            feasible = group[group.mandatory_service_failure_rate <= epsilon + 1e-12]
            feasible_rows.append({
                "configuration": config, "epsilon": epsilon,
                "feasible_cells": len(feasible), "total_cells": len(group),
                "feasible_fraction": len(feasible) / len(group),
                "feasible_mean_dci": feasible.mean_complete_image_dice.mean() if len(feasible) else np.nan,
                "feasible_mean_recall": feasible.pixel_defect_recall.mean() if len(feasible) else np.nan,
            })
    pd.DataFrame(feasible_rows).to_csv(OUT / "hard_realtime_feasible_region.csv", index=False)

    level_rows = []
    for keys, group in data.groupby(["configuration", "period_ms"]):
        config, period = keys
        for subset in ("mandatory", "optional", "all"):
            prefix = "" if subset == "all" else subset + "_"
            counts = np.array([group[f"{prefix}completed_l{level}"].sum() for level in range(1, 6)])
            for level, count in enumerate(counts, start=1):
                level_rows.append({"configuration": config, "period_ms": period,
                                   "subset": subset, "level": level, "count": int(count),
                                   "fraction": count / counts.sum() if counts.sum() else 0.0})
    pd.DataFrame(level_rows).to_csv(OUT / "executed_level_distribution.csv", index=False)

    apply_publication_style()

    # Mutually exclusive terminal outcomes; this is the semantic replacement
    # for the ambiguous mandatory-DMR presentation.
    fig, ax = plt.subplots(figsize=(9.2, 5.15), constrained_layout=True)
    x = np.arange(len(CONFIGS)); bottom = np.zeros(len(CONFIGS))
    stacks = [
        ("mandatory_completed_on_time_rate", "On-time completion", "#4C78A8", "white"),
        ("mandatory_completed_late_rate", "Completed late", "#9B8CC2", "white"),
        ("mandatory_expired_rate", "Expired while waiting", "#D8B365", "black"),
        ("mandatory_dispatch_infeasible_rate", "Dispatch infeasible", "#F2A65A", "black"),
        ("mandatory_allocation_rejected_rate", "Allocation rejected", "#D96C75", "white"),
    ]
    for col, label, color, text_color in stacks:
        values = 100 * summary[col].to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=.42,
               alpha=.92, edgecolor="white", linewidth=.8)
        for xpos, value, base in zip(x, values, bottom):
            if value >= 2.0:
                ax.text(xpos, base + value/2, f"{value:.1f}", ha="center", va="center",
                        fontsize=10.5, fontweight="semibold",
                        color=text_color)
        bottom += values
    ax.set_xticks(x, LABELS); ax.set_ylabel("Released mandatory regions (%)")
    ax.margins(x=.12)
    ax.set_ylim(0, 112); ax.grid(axis="y", linestyle="--", alpha=.25)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.22))
    fig.savefig(OUT / "v10_mandatory_terminal_outcomes.pdf", bbox_inches="tight")
    fig.savefig(OUT / "v10_mandatory_terminal_outcomes.png", bbox_inches="tight")
    plt.close(fig)

    # Dynamic same-network hard-real-time service maps.
    dynamic = ["EDF-Dynamic-Reservation", "HBTASP-Dynamic"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True, sharey=True)
    for ax, config, title in zip(axes, dynamic, ["EDF--Dyn.-Res.", "HBTASP"]):
        pivot = cell[cell.configuration == config].pivot(
            index="lines", columns="period_ms", values="mandatory_service_failure_rate")
        image = ax.imshow(100*pivot.values, aspect="auto", origin="lower", cmap="YlOrRd",
                          vmin=0, vmax=max(1.0, 100*cell[cell.configuration.isin(dynamic)].mandatory_service_failure_rate.max()))
        ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
        ax.set_xlabel("Period / deadline (ms)"); ax.set_title(title, fontsize=11.5)
        for iy in range(len(pivot.index)):
            for ix in range(len(pivot.columns)):
                value = 100*pivot.iloc[iy, ix]
                ax.text(ix, iy, f"{value:.1f}", ha="center", va="center", fontsize=10,
                        color="white" if value > 0.55*100*cell[cell.configuration.isin(dynamic)].mandatory_service_failure_rate.max() else "black")
    axes[0].set_ylabel("Production lines")
    colorbar = fig.colorbar(image, ax=axes, shrink=.9)
    colorbar.set_label("Mandatory-region service failure (%)")
    fig.savefig(OUT / "v10_dynamic_hard_realtime_service_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "v10_dynamic_hard_realtime_service_map.png", bbox_inches="tight")
    plt.close(fig)

    report = {
        "cells": len(data),
        "semantic_identity_max_abs_error": float(
            np.max(np.abs(data.mandatory_dmr-data.mandatory_service_failure_rate))),
        "all_completed_job_dmr_zero": bool((data.mandatory_completed_job_dmr == 0).all()),
        "summary": summary.to_dict(orient="records"),
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
