from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from publication_style import BLUE, GREEN, RED, ORANGE, apply_publication_style


apply_publication_style()

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "perception_evidence/overlap_corrected_final/summary.csv"
out = ROOT / "perception_evidence/overlap_corrected_final/r2_4_overlap_corrected.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(source)
labels = ["Non-overlap\n4x400", "~21-px overlap\n4x416", "100-px overlap\n5x400"]
x = np.arange(3)
colors = [BLUE, GREEN, ORANGE]

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw={"wspace": .32})

w = .26
axes[0].bar(x-w/2, df.micro_dice, w, color=BLUE, label="Micro-Dice",
            alpha=.92, edgecolor="#333333", linewidth=.7)
axes[0].bar(x+w/2, df.micro_recall, w, color=RED, label="Pixel recall",
            alpha=.92, edgecolor="#333333", linewidth=.7)
axes[0].set_ylabel("Score")
axes[0].set_ylim(.74, .82)
axes[0].set_xticks(x, labels)
axes[0].grid(axis="y")
axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=2, frameon=False)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.3f", fontsize=10.5, padding=4,
                      fontweight="semibold")

miss = 100 * df.image_complete_miss_rate
bars = axes[1].bar(x, miss, .46, color=colors, alpha=.92,
                   edgecolor="#333333", linewidth=.7)
axes[1].set_ylabel("Image complete-miss rate (%)")
axes[1].set_ylim(0, max(miss)*1.28)
axes[1].set_xticks(x, labels)
axes[1].grid(axis="y")
axes[1].bar_label(bars, fmt="%.2f", fontsize=10.8, padding=4,
                  fontweight="semibold")

for ax, tag in zip(axes, ["(a)", "(b)"]):
    ax.text(.02, .98, tag, transform=ax.transAxes, va="top", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig(out)
fig.savefig(out.with_suffix(".png"))
print(out)
