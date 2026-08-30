"""Analyze the wide-range cooling audit at mechanism and system levels."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from experiments.publication_style import apply_publication_style, BLUE, ORANGE, GREEN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v18_wide_cooling_mechanism"
KEYS = ["period_ms", "lines", "seed"]
EFFECTS = ["mandatory_rejection_ratio", "admitted_mandatory_violation_ratio",
           "mean_voltage", "peak_modeled_temperature", "total_iit_celsius_seconds",
           "mean_complete_image_dice", "pixel_defect_recall",
           "image_complete_miss_rate"]


def interval(values: pd.Series) -> tuple[float, float, float]:
    mean = float(values.mean())
    sem = stats.sem(values)
    if np.isclose(sem, 0):
        return mean, mean, mean
    lo, hi = stats.t.interval(.95, len(values) - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    expected = 2 * 11 * 4 * 10
    if len(data) != expected:
        raise RuntimeError(f"incomplete grid: {len(data)}/{expected}")
    accounted = (data.completed + data.mandatory_infeasible.fillna(0)
                 + data.optional_skipped.fillna(0)
                 + data.dispatch_infeasible.fillna(0)
                 + data.expired_waiting.fillna(0))
    residual = data.total_regions - accounted
    if np.abs(residual).max() != 0:
        raise RuntimeError("terminal-state accounting is incomplete")
    if data.thermal_violations.sum() != 0:
        raise RuntimeError("unexpected nominal thermal violation")
    off = data[~data.cooling_enabled].set_index(KEYS).sort_index()
    on = data[data.cooling_enabled].set_index(KEYS).sort_index()
    if not off.index.equals(on.index):
        raise RuntimeError("unpaired cells")
    paired = on[EFFECTS].subtract(off[EFFECTS]).reset_index()
    paired.to_csv(OUT / "paired_cell_differences.csv", index=False)

    effect_rows = []
    for period, group in paired.groupby("period_ms"):
        seed_means = group.groupby("seed")[EFFECTS].mean()
        for metric in EFFECTS:
            mean, lo, hi = interval(seed_means[metric])
            effect_rows.append({"period_ms": period, "metric": metric,
                                "cooling_minus_control": mean,
                                "ci95_low": lo, "ci95_high": hi})
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(OUT / "period_effects_seed_clustered.csv", index=False)

    mechanism_rows = []
    enabled = data[data.cooling_enabled]
    for period, group in enabled.groupby("period_ms"):
        for metric in ["cooling_branch_rate", "cooling_locally_decreasing_rate",
                       "cooling_mean_planned_delta_c"]:
            seed_means = group.groupby("seed")[metric].mean()
            mean, lo, hi = interval(seed_means)
            mechanism_rows.append({"period_ms": period, "metric": metric,
                                   "mean": mean, "ci95_low": lo, "ci95_high": hi})
    mechanisms = pd.DataFrame(mechanism_rows)
    mechanisms.to_csv(OUT / "period_mechanism_seed_clustered.csv", index=False)

    all_seed = paired.groupby("seed")[EFFECTS].mean()
    aggregate = []
    for metric in EFFECTS:
        mean, lo, hi = interval(all_seed[metric])
        aggregate.append({"metric": metric, "cooling_minus_control": mean,
                          "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(aggregate).to_csv(OUT / "aggregate_effects_seed_clustered.csv", index=False)

    apply_publication_style()
    periods = sorted(data.period_ms.unique())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    ax = axes[0, 0]
    for metric, label, color in [
        ("cooling_branch_rate", "Low-voltage branch", BLUE),
        ("cooling_locally_decreasing_rate", "Realized modeled temperature decrease", ORANGE),
    ]:
        q = mechanisms[mechanisms.metric == metric].set_index("period_ms").loc[periods]
        ax.plot(periods, q["mean"], marker="o", label=label, color=color)
        ax.fill_between(periods, q.ci95_low, q.ci95_high, color=color, alpha=.14)
    ax.set_ylabel("Observed rate"); ax.legend(frameon=False, loc="upper left")

    for ax, metric, ylabel, color in [
        (axes[0, 1], "mean_voltage", r"$\Delta$ mean voltage (V)", BLUE),
        (axes[1, 0], "peak_modeled_temperature",
         r"$\Delta$ peak modeled temperature ($^\circ$C)", ORANGE),
    ]:
        q = effects[effects.metric == metric].set_index("period_ms").loc[periods]
        ax.plot(periods, q.cooling_minus_control, marker="o", color=color)
        ax.fill_between(periods, q.ci95_low, q.ci95_high, color=color, alpha=.14)
        ax.axhline(0, color="black", linewidth=.8, linestyle="--"); ax.set_ylabel(ylabel)

    ax = axes[1, 1]
    for metric, label, color in [
        ("mean_complete_image_dice", r"$D_{CI}$", BLUE),
        ("pixel_defect_recall", "Recall", GREEN),
        ("image_complete_miss_rate", "Image miss", ORANGE),
    ]:
        q = effects[effects.metric == metric].set_index("period_ms").loc[periods]
        ax.plot(periods, q.cooling_minus_control, marker="o", label=label, color=color)
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_ylabel("Cooling minus control"); ax.legend(frameon=False, loc="best")
    for ax in axes.flat:
        ax.set_xlabel("Period / deadline (ms)"); ax.set_xscale("log")
        ax.set_xticks(periods, [str(x) for x in periods], rotation=35)
        ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "v18_wide_cooling_mechanism.pdf", bbox_inches="tight")
    fig.savefig(OUT / "v18_wide_cooling_mechanism.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    wide = effects.pivot(index="period_ms", columns="metric",
                         values="cooling_minus_control")
    mech = mechanisms.pivot(index="period_ms", columns="metric", values="mean")
    report = ["# Wide-range productive-cooling mechanism audit", "",
              "All effects are cooling minus control and use paired seeds.", "",
              "## Mechanism activation by period", "", "```csv",
              mech.to_csv(), "```", "",
              "## System effects by period", "", "```csv",
              wide.to_csv(), "```", ""]
    (OUT / "ANALYSIS.md").write_text("\n".join(report), encoding="utf-8")
    checkpoint_path = OUT / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["status"] = "complete_validated"
    checkpoint["max_terminal_accounting_residual"] = float(np.abs(residual).max())
    checkpoint["total_nominal_thermal_violations"] = int(data.thermal_violations.sum())
    checkpoint["sha256"].update({
        "runner": sha256(ROOT / "experiments/run_v18_wide_cooling_mechanism.py"),
        "instrumented_event_replay": sha256(
            ROOT / "experiments/v18_instrumented_event_replay.py"),
        "analyzer": sha256(Path(__file__)),
        "results": sha256(OUT / "cell_results.csv"),
    })
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(mech.to_string()); print(wide.to_string())


if __name__ == "__main__":
    main()
