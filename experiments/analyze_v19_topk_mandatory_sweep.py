"""Analyze the Top-K mandatory-region coverage--schedulability sweep."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.publication_style import apply_publication_style, BLUE, GREEN, ORANGE, PURPLE, RED
except ImportError:
    from publication_style import apply_publication_style, BLUE, GREEN, ORANGE, PURPLE, RED


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v19_topk_mandatory_sweep"
RESULT = OUT / "cell_results.csv"
COUNTS = [1, 2, 3, 4]
T_CRIT_95_DF9 = 2.262157

METRICS = [
    "mandatory_service_failure_rate",
    "mandatory_admitted_dmr",
    "optional_region_on_time_coverage",
    "defect_region_mandatory_coverage",
    "defect_region_on_time_coverage",
    "all_defect_regions_on_time_image_rate",
    "any_defect_region_on_time_image_rate",
    "mean_complete_image_dice",
    "pixel_defect_recall",
    "image_complete_miss_rate",
]


def seed_summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(["mandatory_count", "seed"], as_index=False)[METRICS].mean()


def interval_table(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for count, group in seed_level.groupby("mandatory_count"):
        row = {"mandatory_count": int(count)}
        for metric in METRICS:
            values = group[metric].dropna().to_numpy()
            mean = float(values.mean()) if len(values) else float("nan")
            half = (T_CRIT_95_DF9 * values.std(ddof=1) / np.sqrt(len(values))
                    if len(values) > 1 else float("nan"))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mandatory_count")


def plot_metric(ax, table, metric, label, color, marker="o"):
    mean = table[f"{metric}_mean"].to_numpy()
    low = table[f"{metric}_ci95_low"].to_numpy()
    high = table[f"{metric}_ci95_high"].to_numpy()
    ax.errorbar(
        COUNTS, mean, yerr=np.vstack([mean - low, high - mean]),
        marker=marker, color=color, label=label, capsize=3.5,
    )


def main() -> None:
    data = pd.read_csv(RESULT)
    expected = 4 * 5 * 4 * 10
    if len(data) != expected:
        raise RuntimeError(f"expected {expected} cells, found {len(data)}")
    if data.duplicated(["mandatory_count", "period_ms", "lines", "seed"]).any():
        raise RuntimeError("duplicate experiment cells")
    if data["mandatory_terminal_residual"].abs().max() != 0:
        raise RuntimeError("nonzero mandatory terminal residual")

    seeds = seed_summary(data)
    table = interval_table(seeds)
    table.to_csv(OUT / "topk_summary_with_seed_ci.csv", index=False)
    data.groupby(["mandatory_count", "period_ms"], as_index=False)[METRICS].mean().to_csv(
        OUT / "topk_period_summary.csv", index=False
    )

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)

    plot_metric(axes[0, 0], table, "defect_region_mandatory_coverage",
                "Marked mandatory", BLUE, "o")
    plot_metric(axes[0, 0], table, "defect_region_on_time_coverage",
                "Returned on time", ORANGE, "s")
    axes[0, 0].set_title("(a) Defect-positive regional coverage")

    plot_metric(axes[0, 1], table, "mandatory_service_failure_rate",
                "Mandatory service failure", RED, "o")
    plot_metric(axes[0, 1], table, "mandatory_admitted_dmr",
                "Admitted-job DMR", PURPLE, "s")
    axes[0, 1].set_title("(b) Real-time service cost")

    plot_metric(axes[1, 0], table, "mean_complete_image_dice",
                r"$D_{\mathrm{CI}}$", BLUE, "o")
    plot_metric(axes[1, 0], table, "pixel_defect_recall",
                "End-to-end recall", GREEN, "s")
    axes[1, 0].set_title("(c) Delivered perception")

    plot_metric(axes[1, 1], table, "all_defect_regions_on_time_image_rate",
                "All defect regions on time", GREEN, "o")
    plot_metric(axes[1, 1], table, "image_complete_miss_rate",
                "Image complete-miss", RED, "s")
    axes[1, 1].set_title("(d) Image-level outcomes")

    for ax in axes.flat:
        ax.set_xlabel("Mandatory regions per image, $K_M$")
        ax.set_ylabel("Rate")
        ax.set_xticks(COUNTS)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)

    fig.savefig(OUT / "v19_topk_mandatory_tradeoff.pdf")
    fig.savefig(OUT / "v19_topk_mandatory_tradeoff.png", dpi=300)
    plt.close(fig)

    compact = table[[
        "mandatory_count",
        "mandatory_service_failure_rate_mean",
        "defect_region_mandatory_coverage_mean",
        "defect_region_on_time_coverage_mean",
        "all_defect_regions_on_time_image_rate_mean",
        "mean_complete_image_dice_mean",
        "pixel_defect_recall_mean",
        "image_complete_miss_rate_mean",
    ]]
    print(compact.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
