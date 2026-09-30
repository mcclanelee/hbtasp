"""Shared Plot-V9 publication style for all statistical figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

BLUE = "#4C72B0"
GREEN = "#55A868"
RED = "#C44E52"
ORANGE = "#DD8452"
PURPLE = "#8172B2"
GRAY = "#6E6E6E"
THRESHOLD_RED = "#D62728"

METHOD_COLORS = {
    "EDF-FixedL3": BLUE,
    "EDF-Dynamic-Reservation": GREEN,
    "HBTASP-FixedL3": RED,
    "HBTASP-Dynamic": ORANGE,
    "Static-V": BLUE,
    "ESATD-L3": GREEN,
    "HEAT-L3": RED,
    "Full-HBTASP": ORANGE,
}


def apply_publication_style() -> None:
    """Apply a readable Plot-V9/Nature-style statistical-figure theme."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 12.0,
        "axes.labelsize": 13.0,
        "axes.titlesize": 13.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 10.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.1,
        "lines.markersize": 6.5,
        "lines.markeredgewidth": 1.0,
        "errorbar.capsize": 3.5,
        "legend.frameon": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.28,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })
