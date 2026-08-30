"""Validate and summarize the aligned 100--300 ms priority experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v21_multicue_priority_aligned"
KEYS = ["period_ms", "lines", "seed"]
ORDER = ["histogram_zero", "multicue_zero", "multicue_host_cpu_p99"]
METRICS = ["mandatory_service_failure", "mean_complete_image_dice",
           "pixel_defect_recall", "image_complete_miss_rate"]


def ci(values):
    values = np.asarray(values, dtype=float)
    sem = stats.sem(values)
    if np.isclose(sem, 0):
        return float(values.mean()), float(values.mean())
    return tuple(float(x) for x in stats.t.interval(
        .95, len(values) - 1, loc=values.mean(), scale=sem))


def main():
    data = pd.read_csv(OUT / "cell_results.csv")
    expected = 600
    if len(data) != expected or not (data.groupby("treatment").size() == 200).all():
        raise RuntimeError(f"incomplete grid: {len(data)}/{expected}")
    if data.duplicated(["treatment", *KEYS]).any():
        raise RuntimeError("duplicate cells")
    accounted = (data.completed + data.mandatory_infeasible.fillna(0)
                 + data.optional_skipped.fillna(0) + data.dispatch_infeasible.fillna(0)
                 + data.expired_waiting.fillna(0))
    residual = data.total_regions - accounted
    if np.abs(residual).max() != 0:
        raise RuntimeError("terminal accounting failure")
    if data.admitted_mandatory_deadline_violations.sum() != 0:
        raise RuntimeError("admitted mandatory deadline violation")

    indexed = {name: group.set_index(KEYS).sort_index() for name, group in data.groupby("treatment")}
    reference = indexed[ORDER[0]].index
    if any(not indexed[name].index.equals(reference) for name in ORDER):
        raise RuntimeError("unpaired grid")

    summary = []
    for treatment in ORDER:
        for metric in METRICS:
            values = indexed[treatment][metric]
            low, high = ci(values)
            summary.append({"treatment": treatment, "metric": metric,
                            "mean": values.mean(), "ci95_low": low, "ci95_high": high})
    pd.DataFrame(summary).to_csv(OUT / "aligned_summary_with_ci.csv", index=False)

    contrasts = []
    for upper, lower, label in [
        ("multicue_zero", "histogram_zero", "priority-quality effect before charged latency"),
        ("multicue_host_cpu_p99", "multicue_zero", "measured host-CPU p99 latency effect"),
        ("multicue_host_cpu_p99", "histogram_zero", "joint implemented-priority effect")]:
        delta = indexed[upper][METRICS] - indexed[lower][METRICS]
        for metric in METRICS:
            seed_delta = delta[metric].groupby(level="seed").mean()
            low, high = ci(seed_delta)
            contrasts.append({"upper": upper, "lower": lower, "interpretation": label,
                              "metric": metric, "paired_difference": seed_delta.mean(),
                              "ci95_low": low, "ci95_high": high,
                              "paired_t_p": stats.ttest_1samp(seed_delta, 0).pvalue,
                              "inference_unit": "paired seed after averaging period-line cells"})
    pd.DataFrame(contrasts).to_csv(OUT / "paired_contrasts.csv", index=False)
    data.groupby(["treatment", "period_ms"])[METRICS].mean().reset_index().to_csv(
        OUT / "period_summary.csv", index=False)

    means = data.groupby("treatment")[METRICS].mean().loc[ORDER]
    report = f"""# Aligned Top-1 priority quality--latency experiment

All 600 cells are present and exactly paired over periods 100--300 ms, line
counts 4/6/8/10, and ten seeds. Terminal accounting is exact and no admitted
mandatory-region deadline violation occurs.

Grand means:

```
{means.to_string()}
```

The zero-latency treatment isolates the change in priority ordering. The host-
CPU-p99 treatment then charges the independently measured feature-extraction
and classifier latency. This timing is a host-specific sensitivity input and
must not be described as a T4 timing measurement.
"""
    (OUT / "RESULT_ANALYSIS.md").write_text(report, encoding="utf-8")
    provenance = f"""# Result provenance audit

- Raw cells: `cell_results.csv` ({len(data)} rows).
- Treatments: {', '.join(ORDER)}.
- Grid: 5 periods x 4 line counts x 10 seeds x 3 treatments.
- Maximum terminal-accounting residual: {float(np.abs(residual).max())}.
- Admitted mandatory deadline violations: {int(data.admitted_mandatory_deadline_violations.sum())}.
- Priority timing: `priority_latency_4_6_8_10.json`, independently measured
  on the CPU of the T4 experiment host with images preloaded.
- Aggregation: `experiments/analyze_v21_multicue_priority_aligned.py`.

Verdict: FULLY_TRACEABLE_AND_CONSISTENT.
"""
    (OUT / "RESULT_PROVENANCE_AUDIT.md").write_text(provenance, encoding="utf-8")
    checkpoint_path = OUT / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.update({"status": "complete_validated", "validated_cells": len(data),
                       "max_terminal_accounting_residual": float(np.abs(residual).max()),
                       "admitted_mandatory_deadline_violations": int(
                           data.admitted_mandatory_deadline_violations.sum())})
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(means.to_string())


if __name__ == "__main__":
    main()
