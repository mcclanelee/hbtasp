"""Plot authoritative V5 same-network EDF/HBTASP dynamic trade-off."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
try:
    from experiments.publication_style import BLUE, ORANGE
except ImportError:  # direct ``python experiments/script.py`` execution
    from publication_style import BLUE, ORANGE


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial"
DATA = OUT / "cell_results.csv"
METHODS = ("EDF-Dynamic-Reservation", "HBTASP-Dynamic")
LABELS = {"EDF-Dynamic-Reservation": "EDF--Dyn.-Res.",
          "HBTASP-Dynamic": "HBTASP"}
COLORS = {"EDF-Dynamic-Reservation": BLUE, "HBTASP-Dynamic": ORANGE}
MARKERS = {"EDF-Dynamic-Reservation": "s", "HBTASP-Dynamic": "o"}
PANELS = (
    ("mandatory_dmr", "Mandatory rejection rate"),
    ("mean_complete_image_dice", r"Complete-image $D_{CI}$"),
    ("pixel_defect_recall", "Defect recall"),
    ("image_complete_miss_rate", "Image complete-miss rate"),
)


def main():
    data = pd.read_csv(DATA)
    data = data[data.configuration.isin(METHODS)].copy()
    if len(data) != 400:
        raise RuntimeError(f"expected 400 dynamic cells, found {len(data)}")
    periods = sorted(data.period_ms.unique())
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
		"font.size": 10.5, "axes.labelsize": 11.5, "legend.fontsize": 9.2,
		"xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.linewidth": .8,
        "lines.linewidth": 2, "lines.markersize": 7, "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.dpi": 300,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    handles = []
    for ax, (metric, ylabel) in zip(axes.flat, PANELS):
        for method in METHODS:
            means, errors = [], []
            for period in periods:
                group = data[(data.configuration == method) &
                             (data.period_ms == period)]
                # Production-line counts are fixed design cells.  Average them
                # within seed before forming the ten-seed interval stated in
                # the manuscript caption.
                values = group.groupby("seed")[metric].mean().to_numpy()
                means.append(values.mean())
                errors.append(stats.t.ppf(.975, len(values)-1) * stats.sem(values))
            line = ax.errorbar(periods, means, yerr=errors, color=COLORS[method],
                               marker=MARKERS[method], markerfacecolor="white",
                               markeredgewidth=1.2, capsize=3, label=LABELS[method])
            if metric == PANELS[0][0]:
                handles.append(line)
        ax.set_xlabel("Period / deadline (ms)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(periods)
        ax.grid(alpha=.25)
        ax.set_ylim(bottom=0)
    fig.legend(handles=handles, labels=[LABELS[x] for x in METHODS],
               loc="upper center", bbox_to_anchor=(.5, .995), ncol=2,
               frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(rect=(0, 0, 1, .925))
    fig.savefig(OUT / "r2_7_r2_8_calibrated_dynamic_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_7_r2_8_calibrated_dynamic_tradeoff.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for method in METHODS:
        means, errors = [], []
        for period in periods:
            group = data[(data.configuration == method) & (data.period_ms == period)].copy()
            group["optimism_gap"] = (
                group.mean_completed_only_dice - group.mean_complete_image_dice
            )
            values = group.groupby("seed")["optimism_gap"].mean().to_numpy()
            means.append(values.mean())
            errors.append(stats.t.ppf(.975, len(values)-1) * stats.sem(values))
        ax.errorbar(periods, means, yerr=errors, color=COLORS[method],
                    marker=MARKERS[method], markerfacecolor="white",
                    markeredgewidth=1.2, capsize=3, label=LABELS[method])
    ax.set_xlabel("Period / deadline (ms)")
    ax.set_ylabel(r"Optimism gap (completed-only Dice $-D_{CI}$)")
    ax.set_xticks(periods); ax.set_ylim(bottom=0); ax.grid(alpha=.25)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=2,
              frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout()
    fig.savefig(OUT / "r2_8_calibrated_optimism_gap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_8_calibrated_optimism_gap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
