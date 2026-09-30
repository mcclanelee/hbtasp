"""Plot period-by-line-count service boundaries for full-image and HBTASP modes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.publication_style import apply_publication_style
except ImportError:
    from publication_style import apply_publication_style


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "experiments" / "checkpoints"
FULL_PATH = CHECKPOINTS / "v13_full_image_t4_boundary" / "cell_results.csv"
OVERALL_PATH = CHECKPOINTS / "v11_unified_overall" / "cell_results.csv"
STRICT_PATH = CHECKPOINTS / "v14_strict_all_regions_matched" / "cell_results.csv"
OUT = CHECKPOINTS / "v13_full_image_t4_boundary"

PERIODS = [100, 150, 200, 250, 300]
LINES = [4, 6, 8, 10]


def matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    """Return a line-count x period matrix after averaging paired seeds."""
    grouped = frame.groupby(["lines", "period_ms"], as_index=False)[value].mean()
    return (
        grouped.pivot(index="lines", columns="period_ms", values=value)
        .reindex(index=LINES, columns=PERIODS)
        .to_numpy()
    )


def main() -> None:
    full = pd.read_csv(FULL_PATH)
    overall = pd.read_csv(OVERALL_PATH)
    strict = pd.read_csv(STRICT_PATH)

    panels = [
        (
            "(a) DeepLab full image",
            "Complete-image service failure",
            matrix(
                full[full.configuration.eq("DeepLabV3-MobileNetV3-Full")],
                "mandatory_service_failure_rate",
            ),
        ),
        (
            "(b) YOLOv8n full image",
            "Complete-image service failure",
            matrix(
                full[full.configuration.eq("YOLOv8n-Full")],
                "mandatory_service_failure_rate",
            ),
        ),
        (
            "(c) HBTASP partial service",
            "Mandatory-region service failure",
            matrix(
                overall[overall.configuration.eq("Full-HBTASP")],
                "mandatory_service_failure_rate",
            ),
        ),
        (
            "(d) HBTASP all-region",
            "Complete-image service failure",
            1.0 - matrix(strict, "full_image_on_time_acceptance"),
        ),
    ]

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.9), constrained_layout=True)
    images = []
    for ax, (title, endpoint, values) in zip(axes.flat, panels):
        image = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
        images.append(image)
        ax.set_title(title, fontweight="bold", pad=17)
        ax.text(
            0.5,
            1.015,
            endpoint,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.2,
            color="#444444",
        )
        ax.set_xticks(range(len(PERIODS)), PERIODS)
        ax.set_yticks(range(len(LINES)), LINES)
        ax.set_xlabel("Period (ms)")
        ax.set_ylabel("Production lines")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("#B0B0B0")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                color = "white" if value >= 0.48 else "#222222"
                ax.text(
                    col,
                    row,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10.0,
                    fontweight="semibold",
                    color=color,
                )

    colorbar = fig.colorbar(images[0], ax=axes, location="right", shrink=0.88, pad=0.02)
    colorbar.set_label("Service failure rate")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.set_ticklabels([f"{int(x * 100)}%" for x in np.linspace(0.0, 1.0, 6)])

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT / f"v15_full_image_service_boundary_heatmap.{suffix}"
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(path, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
