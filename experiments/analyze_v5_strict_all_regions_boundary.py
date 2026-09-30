"""Validate and plot the all-regions-mandatory schedulability boundary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from experiments.publication_style import apply_publication_style
except ImportError:
    from publication_style import apply_publication_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v5_strict_all_regions_boundary"
PERIODS = [50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
LINES = [4, 5, 6, 7, 8, 9, 10]


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    if len(data) != 700 or not (data.groupby(["period_ms", "lines"]).size() == 10).all():
        raise RuntimeError("strict-boundary grid is incomplete")
    accounted = (data.completed + data.mandatory_infeasible.fillna(0)
                 + data.optional_skipped.fillna(0) + data.dispatch_infeasible.fillna(0)
                 + data.expired_waiting.fillna(0))
    residual = data.total_regions - accounted
    if np.abs(residual).max() != 0:
        raise RuntimeError("terminal accounting failure")
    if data.optional_skipped.sum() != 0:
        raise RuntimeError("strict mode skipped optional work")
    if data.thermal_violations.sum() != 0:
        raise RuntimeError("nominal thermal violation")
    if not np.all(data.total_regions == data.lines * data.epochs * 4):
        raise RuntimeError("strict total-region count mismatch")

    metrics = ["pre_execution_rejection_ratio", "admitted_deadline_violation_ratio",
               "full_image_on_time_acceptance", "mean_complete_image_dice",
               "pixel_defect_recall", "image_complete_miss_rate",
               "peak_modeled_temperature"]
    summary = data.groupby(["period_ms", "lines"])[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "phase_boundary_summary.csv")

    rejection = data.pivot_table(index="lines", columns="period_ms",
                                 values="pre_execution_rejection_ratio", aggfunc="mean").loc[LINES, PERIODS]
    acceptance = data.pivot_table(index="lines", columns="period_ms",
                                  values="full_image_on_time_acceptance", aggfunc="mean").loc[LINES, PERIODS]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    apply_publication_style()
    for ax, matrix, label, cmap in [
        (axes[0], rejection, "Pre-execution rejection ratio", "magma_r"),
        (axes[1], acceptance, "Full-image on-time acceptance", "viridis"),
    ]:
        im = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(PERIODS)), PERIODS, rotation=35)
        ax.set_yticks(range(len(LINES)), LINES)
        ax.set_xlabel("Period / relative deadline (ms)")
        ax.set_ylabel("Production lines")
        for i in range(len(LINES)):
            for j in range(len(PERIODS)):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if value < .25 or value > .72 else "black")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label=label)
    fig.savefig(OUT / "r2_3_strict_admission_phase_boundary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_3_strict_admission_phase_boundary.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    report = f"""# All-regions-mandatory schedulability boundary

All 700 cells are present and terminal accounting is exact. Optional skips are
zero by construction, and nominal modeled thermal violations are
{int(data.thermal_violations.sum())}. Across all cells, the number of deadline
violations after Stage-I admission is
{int(data.admitted_deadline_violations.sum())}; pre-execution rejection is
reported separately and is not renamed as a deadline miss.

This experiment identifies the strict-mode feasibility/acceptance boundary. It
does not claim that zero admitted-job lateness means all demand was accepted:
the heatmap reports the rejected fraction and full-image on-time acceptance
alongside the admitted-job guarantee.
"""
    (OUT / "ANALYSIS.md").write_text(report, encoding="utf-8")
    cp_path = OUT / "checkpoint.json"
    cp = json.loads(cp_path.read_text(encoding="utf-8"))
    cp.update({"status": "complete_validated", "validated_cells": len(data),
               "max_terminal_accounting_residual": float(np.abs(residual).max()),
               "total_optional_skipped": int(data.optional_skipped.sum()),
               "total_thermal_violations": int(data.thermal_violations.sum()),
               "total_admitted_deadline_violations": int(data.admitted_deadline_violations.sum())})
    cp_path.write_text(json.dumps(cp, indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
