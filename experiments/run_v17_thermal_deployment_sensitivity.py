"""Thermal-model mismatch and ambient-temperature sensitivity for HBTASP.

The HBTASP controller keeps the nominal manuscript RC parameters. Only the
replayed plant changes. An effective response factor q changes both the
steady-state rise above nominal ambient and the time constant by q. This is a
structured envelope for the absorbed (alpha, B) model, not identification of a
unique physical thermal resistance or a re-tuned controller sweep.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experiments.initial_manuscript_event_replay import (
    plant_end_temperature,
    plant_steady_temperature,
    run_continuous_hbtasp,
    temperature_excursion_integral,
)
from experiments.initial_manuscript_hbtasp import TAMB, THERMAL_B, VOLTAGES
from experiments.initial_r2_8_metrics import load_confusion, score_trace

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
OUT = ROOT / "experiments/checkpoints/v17_thermal_deployment_sensitivity"
RESULT = OUT / "cell_results.csv"
PERIOD_MS = 200
LINES = 4
EPOCHS = 1000
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
SCENARIOS = (
    ("response", 0.80, 25.0),
    ("response", 0.90, 25.0),
    ("response", 1.00, 25.0),
    ("response", 1.10, 25.0),
    ("response", 1.20, 25.0),
    ("ambient", 1.00, 20.0),
    ("ambient", 1.00, 30.0),
    ("ambient", 1.00, 35.0),
)

_POOL = None
_CONFUSION = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temperature_integral(start: float, voltage: float, duration: float,
                         response_factor: float, ambient_c: float) -> float:
    if duration <= 0:
        return 0.0
    beta = THERMAL_B / response_factor
    tss = plant_steady_temperature(voltage, response_factor, ambient_c)
    return tss * duration + (start - tss) * (1.0 - math.exp(-beta * duration)) / beta


def reconstruct_thermal(trace: list[dict], response_factor: float,
                        ambient_c: float) -> dict[str, float]:
    states = [
        {"temp": ambient_c, "time": 0.0, "integral": 0.0,
         "iit": 0.0, "peak": ambient_c, "active": None}
        for _ in range(2)
    ]
    horizon = max((float(e["time"]) for e in trace if "time" in e), default=0.0)

    def add_segment(state: dict, voltage: float, duration: float) -> None:
        start = state["temp"]
        state["integral"] += temperature_integral(
            start, voltage, duration, response_factor, ambient_c
        )
        state["iit"] += temperature_excursion_integral(
            start, voltage, duration, response_factor=response_factor,
            ambient_c=ambient_c,
        )
        end = plant_end_temperature(
            start, voltage, duration, response_factor, ambient_c
        )
        state["peak"] = max(state["peak"], start, end)
        state["temp"] = end

    for event in trace:
        kind = event.get("event")
        if kind not in ("dispatch", "complete") or "processor" not in event:
            continue
        state = states[int(event["processor"])]
        now = float(event["time"])
        if kind == "dispatch":
            if state["active"] is not None:
                raise RuntimeError("overlapping dispatch on one processor")
            idle = now - state["time"]
            if idle < -1e-12:
                raise RuntimeError("non-monotone trace")
            add_segment(state, VOLTAGES[0], idle)
            state["time"] = now
            if abs(float(event["temperature"]) - state["temp"]) > 2e-8:
                raise RuntimeError("dispatch temperature does not match replayed plant")
            state["active"] = (event["uid"], now, float(event["voltage"]))
        else:
            active = state["active"]
            if active is None or active[0] != event["uid"]:
                raise RuntimeError("completion without matching dispatch")
            add_segment(state, active[2], now - active[1])
            state["time"] = now
            state["active"] = None
            if abs(float(event["temperature"]) - state["temp"]) > 2e-8:
                raise RuntimeError("completion temperature does not match replayed plant")

    for state in states:
        if state["active"] is not None:
            raise RuntimeError("unfinished active segment")
        add_segment(state, VOLTAGES[0], horizon - state["time"])

    return {
        "average_temperature_c": sum(x["integral"] for x in states) / (2 * horizon),
        "iit_celsius_seconds": sum(x["iit"] for x in states),
        "peak_temperature_c": max(x["peak"] for x in states),
        "observation_horizon_s": horizon,
    }


def init_worker() -> None:
    global _POOL, _CONFUSION
    _POOL = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    _CONFUSION = load_confusion(CONFUSION_PATH)


def run_cell(key: tuple[str, float, float, int]) -> dict:
    axis, response_factor, ambient_c, seed = key
    summary, trace = run_continuous_hbtasp(
        _POOL, PERIOD_MS, LINES, EPOCHS, seed,
        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
        plant_thermal_response_factor=response_factor,
        plant_ambient_c=ambient_c,
    )
    quality, _ = score_trace(trace, _POOL, _CONFUSION)
    thermal = reconstruct_thermal(trace, response_factor, ambient_c)
    return {
        "axis": axis, "response_factor": response_factor,
        "ambient_c": ambient_c, "period_ms": PERIOD_MS, "lines": LINES,
        "epochs": EPOCHS, "seed": seed,
        "mandatory_admitted_dmr": summary["mandatory_admitted_dmr"],
        "mandatory_service_failure_rate": summary["mandatory_service_failure_rate"],
        "mandatory_pre_rejection_rate": summary["mandatory_pre_rejection_rate"],
        **quality, **thermal,
    }


def write_rows(rows: list[dict]) -> None:
    tmp = RESULT.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(RESULT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Exact nominal-equivalence gate for the plant extension.
    from experiments.initial_manuscript_hbtasp import end_temperature
    for voltage in VOLTAGES:
        for start in (25.0, 45.0, 60.0):
            for duration in (0.0, 0.01, 0.1, 1.0):
                error = abs(plant_end_temperature(start, voltage, duration)
                            - end_temperature(start, voltage, duration))
                if error > 1e-12:
                    raise RuntimeError(f"nominal plant equivalence failed: {error}")

    keys = [(axis, q, ambient, seed) for axis, q, ambient in SCENARIOS for seed in SEEDS]
    rows = []
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and "response_factor" not in rows[0]:
            rows = []  # invalidate the pre-freeze, physically over-labeled draft
    done = {(x["axis"], float(x["response_factor"]), float(x["ambient_c"]),
             int(x["seed"])) for x in rows}
    pending = [key for key in keys if key not in done]
    started = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = {executor.submit(run_cell, key): key for key in pending}
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda x: (x["axis"], float(x["response_factor"]),
                                     float(x["ambient_c"]), int(x["seed"])))
            write_rows(rows)
            checkpoint = {
                "status": "complete" if len(rows) == len(keys) else "running",
                "completed_cells": len(rows), "total_cells": len(keys),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds_this_run": time.time() - started,
                "protocol": {
                    "period_ms": PERIOD_MS, "lines": LINES, "epochs": EPOCHS,
                    "seeds": SEEDS, "scenarios": SCENARIOS,
                    "controller_model": "nominal manuscript RC parameters",
                    "plant_model": "ambient-shifted absorbed RC; steady-state rise and time constant scaled by q",
                    "identifiability_boundary": "q is effective, not a uniquely identified physical R_th",
                },
                "sha256": {
                    "pool": sha256(POOL_PATH), "confusion": sha256(CONFUSION_PATH),
                    "event_replay": sha256(ROOT / "experiments/initial_manuscript_event_replay.py"),
                    "algorithm": sha256(ROOT / "experiments/initial_manuscript_hbtasp.py"),
                    "results": sha256(RESULT),
                },
            }
            (OUT / "checkpoint.json").write_text(
                json.dumps(checkpoint, indent=2), encoding="utf-8"
            )
            if len(rows) % 10 == 0 or len(rows) == len(keys):
                print(f"[{len(rows)}/{len(keys)}]", flush=True)


if __name__ == "__main__":
    main()
