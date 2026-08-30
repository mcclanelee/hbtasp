"""R2.6: resumable paired robustness experiment for hidden execution overruns."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.initial_manuscript_event_replay import run_continuous_hbtasp

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/initial_histogram_pool_v1/initial_histogram_test_pool.json"
OUT = ROOT / "experiments/checkpoints/v4_r2_6_hidden_overrun"
RESULT = OUT / "cell_results.csv"
PERIODS = (100, 150, 200, 250, 300)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
SCENARIOS = {
    "nominal": (0.0, 0.0, 0.0),
    "rare_hidden_overrun": (0.005, 0.10, 0.70),
}
LINES = 4
EPOCHS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(rows: list[dict]) -> None:
    tmp = RESULT.with_suffix(".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(RESULT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(r["scenario"], int(r["period_ms"]), int(r["seed"])) for r in rows}
    total = len(SCENARIOS) * len(PERIODS) * len(SEEDS)
    for scenario, (probability, minimum, maximum) in SCENARIOS.items():
        for period in PERIODS:
            for seed in SEEDS:
                key = (scenario, period, seed)
                if key in done:
                    continue
                summary, trace = run_continuous_hbtasp(
                    pool, period, LINES, EPOCHS, seed,
                    budget_mode="assigned_level_sensitivity",
                    network_mode="dynamic",
                    overrun_probability=probability,
                    overrun_min_extra=minimum,
                    overrun_max_extra=maximum,
                    overrun_seed=100000 + seed,
                )
                completed = [x for x in trace if x.get("event") == "complete"]
                overruns = [x for x in completed if x.get("overrun_factor", 1.0) > 1.0]
                mandatory_terminal = [x for x in trace if x.get("mandatory") and x.get("event") in {
                    "complete", "mandatory_infeasible", "expired", "dispatch_infeasible"
                }]
                mandatory_miss = sum(
                    x["event"] != "complete" or x.get("deadline_miss", False)
                    for x in mandatory_terminal
                )
                row = {
                    "scenario": scenario, "period_ms": period, "lines": LINES,
                    "epochs": EPOCHS, "seed": seed,
                    "overrun_probability": probability,
                    "overrun_min_extra": minimum, "overrun_max_extra": maximum,
                    **summary,
                    "overrun_count": len(overruns),
                    "overrun_rate_observed": len(overruns) / len(completed),
                    "iit_celsius_seconds": sum(x["iit_celsius_seconds"] for x in completed),
                    "peak_temperature_c": max(x["temperature"] for x in completed),
                    "mandatory_total": len(mandatory_terminal),
                    "mandatory_missed": mandatory_miss,
                    "mandatory_dmr": mandatory_miss / len(mandatory_terminal),
                }
                rows.append(row)
                write_rows(rows)
                done.add(key)
                checkpoint = {
                    "status": "complete" if len(done) == total else "running",
                    "completed_cells": len(done), "total_cells": total,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol": {
                        "periods_ms": PERIODS, "lines": LINES, "epochs": EPOCHS,
                        "seeds": SEEDS, "scenarios": SCENARIOS,
                        "overrun_visibility": "sampled only after the dispatch decision",
                    },
                    "sha256": {"pool": sha256(POOL_PATH), "results": sha256(RESULT)},
                }
                (OUT / "checkpoint.json").write_text(
                    json.dumps(checkpoint, indent=2), encoding="utf-8"
                )
                print(f"[{len(done)}/{total}] {scenario} T={period} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
