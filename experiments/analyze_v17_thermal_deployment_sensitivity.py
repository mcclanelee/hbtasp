"""Summarize and plot the V17 thermal deployment-sensitivity experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from experiments.publication_style import (
    BLUE, ORANGE, RED, THRESHOLD_RED, apply_publication_style,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v17_thermal_deployment_sensitivity"
DATA = OUT / "cell_results.csv"


def aggregate(group: pd.DataFrame) -> pd.Series:
    metrics = [
        "average_temperature_c", "peak_temperature_c", "iit_celsius_seconds",
        "mandatory_admitted_dmr", "mandatory_service_failure_rate",
        "mean_complete_image_dice", "pixel_defect_recall",
        "image_complete_miss_rate",
    ]
    row = {"n_seeds": len(group)}
    critical = stats.t.ppf(0.975, len(group) - 1)
    for metric in metrics:
        values = group[metric].to_numpy(float)
        row[metric] = values.mean()
        row[f"{metric}_ci95"] = critical * values.std(ddof=1) / np.sqrt(len(values))
    return pd.Series(row)


def plot_axis(ax_temp, ax_iit, frame: pd.DataFrame, x: str, xlabel: str) -> None:
    frame = frame.sort_values(x)
    values = frame[x].to_numpy(float)
    avg = frame.average_temperature_c.to_numpy(float)
    peak = frame.peak_temperature_c.to_numpy(float)
    avg_ci = frame.average_temperature_c_ci95.to_numpy(float)
    peak_ci = frame.peak_temperature_c_ci95.to_numpy(float)
    ax_temp.errorbar(values, avg, yerr=avg_ci, color=BLUE, marker="o",
                     linewidth=1.8, capsize=3, label="Time-average")
    ax_temp.errorbar(values, peak, yerr=peak_ci, color=ORANGE, marker="s",
                     linewidth=1.8, capsize=3, label="Peak")
    ax_temp.axhline(60.0, color=THRESHOLD_RED, linestyle="--", linewidth=1.2,
                    label=r"$T_{\max}=60^{\circ}$C")
    ax_temp.set_xlabel(xlabel)
    ax_temp.set_ylabel("Modeled temperature (°C)")
    ax_temp.grid(True)

    bars = ax_iit.bar(values, frame.iit_celsius_seconds, width=(0.055 if x == "response_factor" else 2.2),
                      color=RED, alpha=0.88)
    ax_iit.errorbar(values, frame.iit_celsius_seconds,
                    yerr=frame.iit_celsius_seconds_ci95, fmt="none",
                    color="#333333", capsize=3, linewidth=0.9)
    ax_iit.bar_label(
        bars,
        labels=["0" if value < 5e-7 else f"{value:.3f}"
                for value in frame.iit_celsius_seconds],
        padding=3, fontsize=9,
    )
    ax_iit.set_xlabel(xlabel)
    ax_iit.set_ylabel(r"IIT ($^{\circ}$C·s per cell)")
    ax_iit.set_ylim(0, max(0.12, 1.20 * (frame.iit_celsius_seconds + frame.iit_celsius_seconds_ci95).max()))
    ax_iit.grid(axis="y")


def main() -> None:
    data = pd.read_csv(DATA)
    summary = (
        data.groupby(["axis", "response_factor", "ambient_c"])
        .apply(aggregate)
        .reset_index()
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    response = summary[summary.axis == "response"]
    ambient = summary[summary.axis == "ambient"]
    # Add the shared nominal point to the ambient curve without duplicating runs.
    nominal = response[np.isclose(response.response_factor, 1.0)].copy()
    nominal["axis"] = "ambient"
    ambient = pd.concat([ambient, nominal], ignore_index=True).sort_values("ambient_c")

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    plot_axis(axes[0, 0], axes[1, 0], response,
              "response_factor", r"Effective thermal-response factor $q$")
    plot_axis(axes[0, 1], axes[1, 1], ambient,
              "ambient_c", r"Absorbed-model ambient offset (°C)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=3)
    for index, ax in enumerate(axes.flat):
        ax.text(-0.13, 1.03, f"({chr(97 + index)})", transform=ax.transAxes,
                fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.09, right=0.98,
                        hspace=0.34, wspace=0.24)
    fig.savefig(OUT / "thermal_deployment_sensitivity.pdf")
    fig.savefig(OUT / "thermal_deployment_sensitivity.png")
    plt.close(fig)

    columns = [
        "axis", "response_factor", "ambient_c", "average_temperature_c",
        "peak_temperature_c", "iit_celsius_seconds",
        "mandatory_admitted_dmr", "mandatory_service_failure_rate",
        "mean_complete_image_dice", "pixel_defect_recall",
    ]
    summary[columns].to_csv(OUT / "reporting_table.csv", index=False)


if __name__ == "__main__":
    main()
