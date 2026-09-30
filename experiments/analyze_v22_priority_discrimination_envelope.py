"""Validate and summarize the priority-discrimination envelope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v22_priority_discrimination_envelope"
KEYS = ["period_ms", "lines", "seed"]
ORDER = ["calibrated_histogram", "random_frozen", "defect_oracle"]
METRICS = [
    "mandatory_service_failure",
    "mean_complete_image_dice",
    "pixel_defect_recall",
    "image_complete_miss_rate",
]


def ci95(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sem = stats.sem(values)
    if np.isclose(sem, 0):
        return float(values.mean()), float(values.mean())
    return tuple(float(value) for value in stats.t.interval(
        0.95, len(values) - 1, loc=values.mean(), scale=sem))


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    counts = data.groupby("strategy").size()
    if len(data) != 600 or set(counts.index) != set(ORDER) or not (counts == 200).all():
        raise RuntimeError(f"incomplete grid: {counts.to_dict()}")
    if data.duplicated(["strategy", *KEYS]).any():
        raise RuntimeError("duplicate strategy-period-lines-seed cells")

    accounted = (data.completed + data.mandatory_infeasible.fillna(0)
                 + data.optional_skipped.fillna(0) + data.dispatch_infeasible.fillna(0)
                 + data.expired_waiting.fillna(0))
    residual = data.total_regions - accounted
    if np.abs(residual).max() != 0:
        raise RuntimeError("terminal accounting failure")
    if data.admitted_mandatory_deadline_violations.sum() != 0:
        raise RuntimeError("admitted mandatory-region deadline violation")

    indexed = {name: group.set_index(KEYS).sort_index()
               for name, group in data.groupby("strategy")}
    reference = indexed[ORDER[0]].index
    if any(not indexed[name].index.equals(reference) for name in ORDER):
        raise RuntimeError("unpaired strategy grids")

    seed_means = data.groupby(["strategy", "seed"])[METRICS].mean().reset_index()
    summary_rows = []
    for strategy in ORDER:
        group = seed_means[seed_means.strategy == strategy]
        for metric in METRICS:
            low, high = ci95(group[metric])
            summary_rows.append({
                "strategy": strategy,
                "metric": metric,
                "mean": group[metric].mean(),
                "ci95_low": low,
                "ci95_high": high,
                "inference_unit": "seed after averaging 20 period-line cells",
                "n": len(group),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "strategy_summary_seed_ci.csv", index=False)

    contrast_rows = []
    base = seed_means[seed_means.strategy == "calibrated_histogram"].set_index("seed")
    for strategy in ("random_frozen", "defect_oracle"):
        other = seed_means[seed_means.strategy == strategy].set_index("seed")
        for metric in METRICS:
            delta = other[metric] - base[metric]
            low, high = ci95(delta)
            contrast_rows.append({
                "upper": strategy,
                "lower": "calibrated_histogram",
                "metric": metric,
                "paired_difference": delta.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "paired_t_p": 1.0 if np.isclose(delta.std(ddof=1), 0)
                else stats.ttest_1samp(delta, 0).pvalue,
                "inference_unit": "paired seed after averaging 20 period-line cells",
                "n": len(delta),
            })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(OUT / "paired_contrasts_seed_ci.csv", index=False)
    data.groupby(["strategy", "period_ms"])[METRICS].mean().reset_index().to_csv(
        OUT / "period_summary.csv", index=False)

    diagnostics = json.loads((OUT / "priority_diagnostics.json").read_text(encoding="utf-8"))
    means = summary.pivot(index="strategy", columns="metric", values="mean").loc[ORDER]
    deltas = contrasts.set_index(["upper", "metric"])
    random_recall = deltas.loc[("random_frozen", "pixel_defect_recall"), "paired_difference"]
    random_dice = deltas.loc[("random_frozen", "mean_complete_image_dice"), "paired_difference"]
    oracle_recall = deltas.loc[("defect_oracle", "pixel_defect_recall"), "paired_difference"]
    oracle_dice = deltas.loc[("defect_oracle", "mean_complete_image_dice"), "paired_difference"]
    report = f"""# Priority-discrimination envelope on the principal grid

All 600 cells are present and paired over five periods, four production-line
counts, and ten seeds. Terminal accounting is exact, and no admitted mandatory
region finishes late. The calibrated-histogram cells are the validated v21
cells; frozen-random and ground-truth-informed diagnostic cells use the same
HBTASP kernel and paired grid.

Priority diagnostics on the frozen 500-image pool:

- calibrated histogram: hit rate {diagnostics['calibrated_histogram']['priority_defect_hit_rate']:.2%}, defect-pixel share {diagnostics['calibrated_histogram']['priority_defect_pixel_share']:.2%};
- frozen random ranking: hit rate {diagnostics['random_frozen']['priority_defect_hit_rate']:.2%}, defect-pixel share {diagnostics['random_frozen']['priority_defect_pixel_share']:.2%};
- ground-truth-informed diagnostic reference: hit rate {diagnostics['defect_oracle']['priority_defect_hit_rate']:.2%}, defect-pixel share {diagnostics['defect_oracle']['priority_defect_pixel_share']:.2%}.

Grand means (seed is the inference unit):

```
{means.to_string()}
```

Relative to the calibrated histogram, frozen random ranking changes
complete-image Dice by {random_dice:+.4f} and pixel recall by {random_recall:+.4f}.
The ground-truth-informed diagnostic reference changes complete-image Dice by {oracle_dice:+.4f} and recall by
{oracle_recall:+.4f}. Because all policies protect exactly one region and add
zero priority-computation latency, mandatory-service failure mainly measures
capacity, while Dice, recall, and image miss expose the delivered-perception
effect of ranking quality. The diagnostic-reference gap quantifies the
remaining improvement opportunity for lightweight deployable estimators.
"""
    (OUT / "RESULT_ANALYSIS.md").write_text(report, encoding="utf-8")

    checkpoint_path = OUT / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.update({
        "status": "complete_validated",
        "validated_cells": len(data),
        "max_terminal_accounting_residual": float(np.abs(residual).max()),
        "admitted_mandatory_deadline_violations": int(
            data.admitted_mandatory_deadline_violations.sum()),
        "inference_unit": "paired seed after averaging period-line cells",
    })
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(means.to_string())
    print("\nPaired contrasts:\n", contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
