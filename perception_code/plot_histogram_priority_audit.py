from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "axes.linewidth": .8, "pdf.fonttype": 42})
    frame = pd.read_csv("histogram_priority_corrected/histogram_priority_summary.csv")
    order = ["original", "bright+20", "bright-20", "contrast+20%", "contrast-20%"]
    labels = ["Original", "Brightness\n+20", "Brightness\n−20", "Contrast\n+20%", "Contrast\n−20%"]
    frame = frame.set_index("mode").loc[order]
    x = np.arange(len(frame)); width = .36
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.bar(x - width/2, frame.roc_auc, width, label="Region ROC-AUC", color="#0072B2")
    ax.bar(x + width/2, frame.top1_defective_region_hit_rate, width,
           label="Top-1 defective-region hit rate", color="#D55E00")
    ax.axhline(.5, color="#CC0000", ls="--", lw=1, label="Random ROC-AUC")
    ax.set_xticks(x, labels); ax.set_ylabel("Score / rate"); ax.set_ylim(0, .72)
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=3, frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)
    fig.tight_layout(rect=(0, 0, 1, .91))
    out = Path("histogram_priority_corrected");
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"r2_5_histogram_priority_audit.{suffix}", dpi=400,
                    bbox_inches="tight")


if __name__ == "__main__":
    main()
