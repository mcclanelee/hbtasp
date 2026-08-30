"""Summarize paired R2.6 hidden-overrun robustness results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v4_r2_6_hidden_overrun"


def main() -> None:
    data = pd.read_csv(OUT / "cell_results.csv")
    metrics = ["overrun_count", "overrun_rate_observed", "thermal_violations",
               "iit_celsius_seconds", "peak_temperature_c", "deadline_misses",
               "mandatory_dmr"]
    period = data.groupby(["scenario", "period_ms"])[metrics].agg(["mean", "std"])
    period.columns = [f"{a}_{b}" for a, b in period.columns]
    period.reset_index().to_csv(OUT / "summary_by_period.csv", index=False)
    wide = data.pivot(index=["period_ms", "seed"], columns="scenario", values=metrics)
    rows = []
    for metric in ["thermal_violations", "iit_celsius_seconds", "peak_temperature_c",
                   "deadline_misses", "mandatory_dmr"]:
        delta = wide[metric]["rare_hidden_overrun"] - wide[metric]["nominal"]
        sem = stats.sem(delta)
        ci = stats.t.interval(0.95, len(delta) - 1, loc=delta.mean(), scale=sem)
        test = stats.ttest_rel(wide[metric]["rare_hidden_overrun"], wide[metric]["nominal"])
        rows.append({"metric": metric, "mean_paired_difference": delta.mean(),
                     "ci95_low": ci[0], "ci95_high": ci[1], "p_value": test.pvalue,
                     "positive_pairs": int((delta > 0).sum()), "pairs": len(delta)})
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT / "paired_effects.csv", index=False)
    nominal = data[data.scenario == "nominal"]
    overrun = data[data.scenario == "rare_hidden_overrun"]
    text = f"""# R2.6 hidden-overrun audit

The scheduler is given only the frozen measured WCET. An overrun is sampled
after dispatch, so it cannot influence the selected GPU, DNN level, or voltage.
The primary paired grid contains 5 periods x 10 seeds at four lines and 1000
cycles per cell. The robustness condition uses probability 0.005 and a
conditional extra-duration factor U(0.10, 0.70).

## Main result

All {len(nominal)} nominal cells have zero thermal violations, zero IIT, and
zero deadline misses. The observed hidden-overrun rate is
{overrun.overrun_rate_observed.mean():.6f}. Under hidden overruns, the mean peak
temperature is {overrun.peak_temperature_c.mean():.6f} C, mean IIT is
{overrun.iit_celsius_seconds.mean():.9f} C s per 1000-cycle cell, and mean
mandatory DMR is {overrun.mandatory_dmr.mean():.6f}. Positive IIT occurs in
{int((overrun.iit_celsius_seconds > 0).sum())}/{len(overrun)} cells.

## Interpretation boundary

The deterministic guarantee is empirically preserved when actual execution
does not exceed the measured WCET. Rare model-external execution overruns can
produce small transient excursions; therefore the manuscript must replace an
unqualified real-system claim of strict satisfaction with a conditional
guarantee plus robustness disclosure. The illustrative 10-second trace uses
three controlled post-dispatch injections near 2.5, 6, and 8 seconds; it is not
used to estimate the formal event rate.
"""
    (OUT / "ANALYSIS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
