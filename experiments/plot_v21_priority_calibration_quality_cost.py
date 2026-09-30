"""Recreate Supplementary Fig. 16 from frozen calibration and v21 summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    ROOT / "perception_evidence/histogram_direction_calibration/test_summary.csv"
)
ALIGNED = (
    ROOT
    / "experiments/checkpoints/v21_multicue_priority_aligned/aligned_summary_with_ci.csv"
)
OUTPUT = ROOT / "experiments/checkpoints/v21_multicue_priority_aligned"


def main() -> None:
    calibration = pd.read_csv(CALIBRATION).set_index("mode").loc[
        ["original", "bright+20", "bright-20", "contrast+20%", "contrast-20%"]
    ]
    calibrated = calibration.roc_auc.to_numpy()
    uncalibrated = 1 - calibrated
    x = np.arange(5)
    width = 0.31

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
         "axes.linewidth": 0.8}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(9.8, 4.05), gridspec_kw={"width_ratios": [1.05, 1.25]}
    )
    axis = axes[0]
    raw_bars = axis.bar(
        x - width / 2, uncalibrated, width, color="#4C78A8",
        label="Uncalibrated direction"
    )
    calibrated_bars = axis.bar(
        x + width / 2, calibrated, width, color="#59A14F",
        label="Validation-calibrated"
    )
    random_line = axis.axhline(
        0.5, color="0.35", linestyle="--", linewidth=1, label="Random ranking"
    )
    axis.set_xticks(
        x, ["Original", "Brightness\n+20", "Brightness\n−20",
            "Contrast\n+20%", "Contrast\n−20%"]
    )
    axis.set_ylabel("Region-level ROC-AUC")
    axis.set_ylim(0.28, 0.735)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(raw_bars, fmt="%.4f", padding=2, fontsize=7.4, rotation=90)
    axis.bar_label(calibrated_bars, fmt="%.4f", padding=2, fontsize=7.4, rotation=90)
    axis.set_title("(a) Held-out direction calibration", loc="left", fontweight="bold")

    aligned = pd.read_csv(ALIGNED)
    metrics = [
        "mandatory_service_failure",
        "mean_complete_image_dice",
        "pixel_defect_recall",
        "image_complete_miss_rate",
    ]
    treatments = ["histogram_zero", "multicue_zero", "multicue_host_cpu_p99"]
    labels = [
        "Calibrated histogram",
        "Multi-cue, cost excluded",
        "Multi-cue, p99 cost included",
    ]
    colors = ["#59A14F", "#F28E2B", "#E15759"]
    positions = np.arange(4)
    bar_width = 0.22
    axis = axes[1]
    legend_handles = []
    for index, (treatment, label, color) in enumerate(zip(treatments, labels, colors)):
        values = [
            float(
                aligned[(aligned.treatment == treatment) & (aligned.metric == metric)]
                .iloc[0]["mean"]
            )
            for metric in metrics
        ]
        bars = axis.bar(
            positions + (index - 1) * bar_width,
            values,
            bar_width,
            color=color,
            label=label,
        )
        legend_handles.append(bars[0])
        axis.bar_label(bars, fmt="%.4f", padding=2, fontsize=7.4, rotation=90)
    axis.set_xticks(
        positions,
        ["Mandatory-region\nservice failure", r"$D_{CI}$", "Pixel recall", "Image miss"],
    )
    axis.set_ylabel("End-to-end metric")
    axis.set_ylim(0, 0.64)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("(b) Priority quality and charged cost", loc="left", fontweight="bold")

    fig.legend(
        [raw_bars[0], calibrated_bars[0], random_line],
        ["Uncalibrated direction", "Validation-calibrated", "Random ranking"],
        frameon=False,
        fontsize=7.2,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.265, 0.018),
        columnspacing=0.9,
        handlelength=1.4,
    )
    fig.legend(
        legend_handles,
        labels,
        frameon=False,
        fontsize=7.2,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.745, 0.018),
        columnspacing=0.9,
        handlelength=1.4,
    )
    fig.tight_layout(w_pad=2.0, rect=(0, 0.15, 1, 1))
    fig.savefig(OUTPUT / "v21_priority_calibration_quality_cost.pdf", bbox_inches="tight")
    fig.savefig(
        OUTPUT / "v21_priority_calibration_quality_cost.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
