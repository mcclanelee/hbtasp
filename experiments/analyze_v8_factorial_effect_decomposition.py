"""Seed-clustered effect decomposition for the audited 2x2 experiment.

This script does not rerun the scheduler.  It analyzes the 800 already-audited
cells in v8_calibrated_final_factorial/cell_results.csv.  The internal
configuration keys are retained for provenance, while publication-facing
labels use HBTASP for the complete method and HBTASP (Fixed L3) for the
restricted network-policy configuration.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/cell_results.csv"
OUT = ROOT / "experiments/checkpoints/v8_factorial_effect_decomposition"

CONFIGS = {
    "ef": "EDF-FixedL3",
    "ed": "EDF-Dynamic-Reservation",
    "hf": "HBTASP-FixedL3",
    "hd": "HBTASP-Dynamic",
}

DISPLAY = {
    "ef": "EDF--FixedL3",
    "ed": "EDF--Dynamic-Reservation",
    "hf": "HBTASP (Fixed L3)",
    "hd": "HBTASP",
}

METRICS = {
    "mandatory_dmr": "Mandatory service failure",
    "mean_complete_image_dice": "Complete-image Dice",
    "pixel_defect_recall": "End-to-end recall",
    "image_complete_miss_rate": "Image complete-miss rate",
}


def t_interval(values: pd.Series) -> Tuple[float, float, float]:
    values = values.astype(float)
    mean = float(values.mean())
    if len(values) < 2 or np.isclose(values.std(ddof=1), 0.0):
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)

    expected = set(CONFIGS.values())
    observed = set(frame["configuration"].unique())
    if observed != expected:
        raise RuntimeError(f"Unexpected configuration set: {sorted(observed)}")

    key = ["period_ms", "lines", "seed"]
    counts = frame.groupby("configuration").size()
    if not (counts == 200).all():
        raise RuntimeError(f"Expected 200 cells per configuration, got {counts.to_dict()}")
    if frame.duplicated(["configuration", *key]).any():
        raise RuntimeError("Duplicate configuration-period-lines-seed cells detected")

    wide = frame.pivot(index=key, columns="configuration", values=list(METRICS))
    if wide.isna().any().any() or len(wide) != 200:
        raise RuntimeError("The 2x2 grid is not completely paired")

    # Each seed is an independent stochastic unit.  Period and line-count cells
    # are fixed operating conditions and are averaged within seed before CI
    # construction, preventing the 200 cells from being treated as independent.
    seed_means = frame.groupby(["seed", "configuration"], as_index=False)[list(METRICS)].mean()
    seed_wide = seed_means.pivot(index="seed", columns="configuration", values=list(METRICS))

    contrast_definitions = [
        (
            "Scheduling effect under Fixed L3",
            lambda m: seed_wide[m][CONFIGS["hf"]] - seed_wide[m][CONFIGS["ef"]],
            f'{DISPLAY["hf"]} minus {DISPLAY["ef"]}',
        ),
        (
            "Scheduling effect under Dynamic L1--L5",
            lambda m: seed_wide[m][CONFIGS["hd"]] - seed_wide[m][CONFIGS["ed"]],
            f'{DISPLAY["hd"]} minus {DISPLAY["ed"]}',
        ),
        (
            "Dynamic-path effect under EDF",
            lambda m: seed_wide[m][CONFIGS["ed"]] - seed_wide[m][CONFIGS["ef"]],
            f'{DISPLAY["ed"]} minus {DISPLAY["ef"]}',
        ),
        (
            "Dynamic-path effect under HBTASP",
            lambda m: seed_wide[m][CONFIGS["hd"]] - seed_wide[m][CONFIGS["hf"]],
            f'{DISPLAY["hd"]} minus {DISPLAY["hf"]}',
        ),
        (
            "Scheduler--path interaction",
            lambda m: (
                seed_wide[m][CONFIGS["hd"]] - seed_wide[m][CONFIGS["hf"]]
                - seed_wide[m][CONFIGS["ed"]] + seed_wide[m][CONFIGS["ef"]]
            ),
            "(HBTASP dynamic-path effect) minus (EDF dynamic-path effect)",
        ),
    ]

    rows = []
    for effect, function, definition in contrast_definitions:
        for metric, metric_label in METRICS.items():
            delta = function(metric)
            mean, low, high = t_interval(delta)
            rows.append(
                {
                    "effect": effect,
                    "definition": definition,
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "independent_seed_units": len(delta),
                }
            )
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT / "seed_clustered_factorial_effects.csv", index=False)

    means = frame.groupby("configuration", as_index=False)[list(METRICS)].mean()
    means["publication_label"] = means["configuration"].map(
        {value: DISPLAY[key_] for key_, value in CONFIGS.items()}
    )
    means.to_csv(OUT / "configuration_means.csv", index=False)

    lines = [
        "# Audited 2x2 scheduler--network effect decomposition",
        "",
        f"Source: `{SOURCE.relative_to(ROOT.parent)}`",
        "",
        "The analysis uses all 800 audited cells (200 per configuration). The ten paired seeds are the independent inferential units; the five periods and four line counts are averaged within each seed before forming Student-t 95% confidence intervals.",
        "",
        "A negative difference favors the first-named configuration for mandatory service failure and image complete-miss rate. A positive difference favors it for complete-image Dice and recall.",
        "",
        "| Effect | Metric | Mean difference | Seed-clustered 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        lines.append(
            f"| {row.effect} | {row.metric_label} | {row.mean_difference:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The crossed design separates the effect of access to Dynamic L1--L5 paths from the effect of replacing the EDF reservation control with the integrated HBTASP scheduling policy. It does not by itself identify the contribution of each internal HBTASP component; Static-V and the existing cooling ablations serve that separate purpose.",
        ]
    )
    (OUT / "FACTORIAL_EFFECT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
