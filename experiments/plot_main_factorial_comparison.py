"""Recreate the main-paper crossed-comparison figure from frozen v8 cells."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.publication_style import apply_publication_style


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/cell_results.csv"
OUTPUT = ROOT / "experiments/checkpoints/v9_restored_main_figures"
CONFIGS = [
    "EDF-FixedL3",
    "EDF-Dynamic-Reservation",
    "HBTASP-FixedL3",
    "HBTASP-Dynamic",
]
LABELS = ["EDF--Fixed L3", "EDF--Dyn.-Res.", "HBTASP--Fixed L3", "HBTASP"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]


def seed_summary(data: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    means, half_widths = [], []
    for configuration in CONFIGS:
        values = data[data.configuration == configuration].groupby("seed")[metric].mean()
        means.append(values.mean())
        sem = stats.sem(values)
        if np.isclose(sem, 0):
            half_widths.append(0.0)
        else:
            low, high = stats.t.interval(
                0.95, len(values) - 1, loc=values.mean(), scale=sem
            )
            half_widths.append((high - low) / 2)
    return np.asarray(means), np.asarray(half_widths)


def main() -> None:
    data = pd.read_csv(INPUT)
    expected = len(CONFIGS) * 5 * 4 * 10
    if len(data) != expected or set(data.configuration) != set(CONFIGS):
        raise RuntimeError("the frozen v8 crossed-comparison grid is incomplete")

    apply_publication_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panels = [
        ("mandatory_dmr", "Mandatory service failure (%)", 100.0),
        ("mean_complete_image_dice", r"$D_{\mathrm{CI}}$", 1.0),
        ("pixel_defect_recall", "Pixel recall", 1.0),
        ("image_complete_miss_rate", "Image miss (%)", 100.0),
    ]
    x = np.arange(len(CONFIGS))
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.55), constrained_layout=True)
    for panel, (axis, (metric, ylabel, scale)) in enumerate(zip(axes, panels)):
        means, half_widths = seed_summary(data, metric)
        means, half_widths = scale * means, scale * half_widths
        bars = axis.bar(
            x,
            means,
            yerr=half_widths,
            capsize=4,
            width=0.62,
            color=COLORS,
            alpha=0.92,
            edgecolor="#333333",
            linewidth=0.75,
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [label.replace("--", "\n") for label in LABELS])
        axis.grid(axis="y", linestyle="--", alpha=0.28)
        axis.text(-0.16, 1.03, f"({chr(97 + panel)})", transform=axis.transAxes,
                  fontweight="bold", va="bottom")
        span = max(means.max() - means.min(), means.max() * 0.15, 0.02)
        lower = 0 if metric in ("mandatory_dmr", "image_complete_miss_rate") else max(
            0, means.min() - 0.45 * span
        )
        axis.set_ylim(lower, means.max() + 0.70 * span)
        for bar, value in zip(bars, means):
            label = f"{value:.2f}" if scale == 100 else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.06 * span,
                label,
                ha="center",
                va="bottom",
                fontsize=10.8,
                fontweight="semibold",
            )
    fig.savefig(OUTPUT / "v9_factorial_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "v9_factorial_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
