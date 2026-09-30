"""HEAT-L3 100-ms feasibility-domain audit.

This experiment does not replace the Overall 4/6/8/10-line grid.  It extends
the 100-ms slice to 1--4 lines so that accepted HEAT cells have defined thermal
trajectories and the transition to Algorithm-2 task-set rejection is visible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments import run_v13_heat_paper_aligned as heat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v15_heat_100ms_feasible_domain"
RESULT = OUT / "cell_results.csv"
PERIOD_MS = 100
LEVEL = 3
LINES = (1, 2, 3, 4)
SEEDS = heat.SEEDS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    heat.init_worker()
    rows = [heat.run_cell(LEVEL, PERIOD_MS, lines, seed)
            for lines in LINES for seed in SEEDS]
    frame = pd.DataFrame(rows).sort_values(["lines", "seed"])
    frame.to_csv(RESULT, index=False)

    accepted = frame[frame.task_set_accepted.astype(bool)].copy()
    summary = frame.groupby("lines", as_index=False).agg(
        accepted_cells=("task_set_accepted", "sum"),
        task_set_acceptance=("task_set_accepted", "mean"),
        failure_endpoint=("mandatory_service_failure_rate", "mean"),
        complete_image_dice=("mean_complete_image_dice", "mean"),
        recall=("pixel_defect_recall", "mean"),
    )
    thermal = accepted.groupby("lines", as_index=False).agg(
        accepted_cell_atp=("average_temperature_c", "mean"),
        accepted_cell_iit=("iit_celsius_seconds", "mean"),
        maximum_peak=("peak_temperature_c", "max"),
    )
    summary = summary.merge(thermal, on="lines", how="left")
    summary.to_csv(OUT / "summary.csv", index=False)

    checkpoint = {
        "status": "complete",
        "purpose": "100-ms HEAT-L3 feasibility-domain extension; not pooled into Overall",
        "period_ms": PERIOD_MS,
        "level": LEVEL,
        "lines": list(LINES),
        "seeds": list(SEEDS),
        "cells": len(frame),
        "completed_cells": len(frame),
        "accepted_cells": int(frame.task_set_accepted.sum()),
        "rejected_cells_with_finite_thermal": int(
            ((~frame.task_set_accepted.astype(bool)) & frame.average_temperature_c.notna()).sum()
        ),
        "sha256": {
            "runner": sha256(Path(__file__)),
            "base_heat_runner": sha256(Path(heat.__file__)),
            "results": sha256(RESULT),
        },
    }
    (OUT / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
