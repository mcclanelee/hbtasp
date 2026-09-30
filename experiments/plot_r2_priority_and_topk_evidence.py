"""Create the Round-2 supplementary figures from frozen replay summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOPK = ROOT / "experiments/checkpoints/v19_topk_mandatory_sweep"
PRIORITY = ROOT / "experiments/checkpoints/v22_priority_discrimination_envelope"


def heatmap(ax, values, rows, columns, title, fmt, cmap, vmin=None, vmax=None):
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xlabel("Release period (ms)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            rgba = image.cmap(image.norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=8, color="black" if luminance > 0.55 else "white")
    ax.set_title(title, fontsize=10, pad=8)
    return image


def pivot(frame, index, column, value, row_order, col_order):
    return (frame.pivot(index=index, columns=column, values=value)
            .reindex(index=row_order, columns=col_order).to_numpy(dtype=float))


def make_topk_figure():
    data = pd.read_csv(TOPK / "topk_period_summary.csv")
    ks = [1, 2, 3, 4]
    periods = [100, 150, 200, 250, 300]
    specs = [
        ("mandatory_service_failure_rate", "(a) Mandatory-service failure", ".1%", "magma_r", 0, 0.8),
        ("defect_region_on_time_coverage", "(b) On-time defect-region coverage", ".1%", "viridis", 0, 1),
        ("mean_complete_image_dice", r"(c) Complete-image Dice $D_{CI}$", ".3f", "viridis", 0.2, 0.55),
        ("pixel_defect_recall", "(d) Pixel recall", ".3f", "viridis", 0.25, 0.7),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for ax, (metric, title, fmt, cmap, vmin, vmax) in zip(axes.flat, specs):
        values = pivot(data, "mandatory_count", "period_ms", metric, ks, periods)
        image = heatmap(ax, values, [f"Top-{k}" for k in ks], periods,
                        title, fmt, cmap, vmin, vmax)
        ax.set_ylabel(r"Mandatory count $K_M$")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Load-dependent Top-$K_M$ service--perception envelope", fontsize=12)
    # Keep the load-stratified Supplementary Fig. 19 separate from the
    # aggregate Top-K trade-off in Supplementary Fig. 14.  The two figures
    # intentionally use the same frozen v19 cells but present different
    # aggregations.
    fig.savefig(TOPK / "v19_topk_load_envelope.pdf", bbox_inches="tight")
    fig.savefig(TOPK / "v19_topk_load_envelope.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_priority_figure():
    data = pd.read_csv(PRIORITY / "period_summary.csv")
    order = ["random_frozen", "calibrated_histogram", "defect_oracle"]
    labels = ["Frozen random", "Calibrated histogram", "GT-informed diagnostic"]
    periods = [100, 150, 200, 250, 300]
    specs = [
        ("mean_complete_image_dice", r"(a) Complete-image Dice $D_{CI}$", ".3f", 0.2, 0.55),
        ("pixel_defect_recall", "(b) Pixel recall", ".3f", 0.25, 0.7),
        ("image_complete_miss_rate", "(c) Image complete-miss rate", ".1%", 0, 0.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.65), constrained_layout=True)
    for ax, (metric, title, fmt, vmin, vmax) in zip(axes, specs):
        values = pivot(data, "strategy", "period_ms", metric, order, periods)
        cmap = "magma_r" if metric == "image_complete_miss_rate" else "viridis"
        image = heatmap(ax, values, labels, periods, title, fmt, cmap, vmin, vmax)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Priority-discrimination envelope at zero ranking latency", fontsize=12)
    fig.savefig(PRIORITY / "v22_priority_discrimination_envelope.pdf", bbox_inches="tight")
    fig.savefig(PRIORITY / "v22_priority_discrimination_envelope.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_topk_figure()
    make_priority_figure()
