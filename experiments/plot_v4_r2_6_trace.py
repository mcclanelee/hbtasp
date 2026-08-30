"""Generate the R2.6 10-second illustrative hidden-overrun trace."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    from publication_style import BLUE, ORANGE, GREEN, THRESHOLD_RED, apply_publication_style
except ImportError:  # support ``python -m experiments.plot_v4_r2_6_trace``
    from experiments.publication_style import (BLUE, ORANGE, GREEN, THRESHOLD_RED,
                                               apply_publication_style)

from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_manuscript_hbtasp import TAMB, VOLTAGES, end_temperature

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "experiments/checkpoints/initial_histogram_pool_v1/initial_histogram_test_pool.json"
OUT = ROOT / "experiments/checkpoints/v4_r2_6_hidden_overrun"

apply_publication_style()


def reconstruct(trace: list[dict], processor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dispatches = [x for x in trace if x.get("event") == "dispatch" and x["processor"] == processor]
    grid = np.linspace(0.0, 10.0, 10001)
    temperature = np.empty_like(grid)
    voltage = np.full_like(grid, VOLTAGES[0])
    cursor_time, cursor_temp = 0.0, TAMB
    for event in dispatches:
        start = event["time"]
        stop = min(10.0, start + event["planned_duration"] * event["overrun_factor"])
        idle = (grid >= cursor_time) & (grid < start)
        temperature[idle] = [end_temperature(cursor_temp, VOLTAGES[0], t - cursor_time) for t in grid[idle]]
        cursor_temp = end_temperature(cursor_temp, VOLTAGES[0], max(0.0, start - cursor_time))
        active = (grid >= start) & (grid < stop)
        v = event["voltage"]
        temperature[active] = [end_temperature(cursor_temp, v, t - start) for t in grid[active]]
        voltage[active] = v
        cursor_temp = end_temperature(cursor_temp, v, max(0.0, stop - start))
        cursor_time = stop
    tail = grid >= cursor_time
    temperature[tail] = [end_temperature(cursor_temp, VOLTAGES[0], t - cursor_time) for t in grid[tail]]
    return grid, temperature, voltage


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    summary, trace = run_continuous_hbtasp(
        pool, 100, 4, 100, 19,
        budget_mode="assigned_level_sensitivity", network_mode="dynamic",
        forced_overrun_times=(2.5, 6.0, 8.0), forced_overrun_extra=0.70,
        forced_overrun_min_voltage=0.85,
        release_mode="poisson",
    )
    records = []
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.4), sharex=True)
    colors = (ORANGE, BLUE)
    for processor, ax in enumerate(axes):
        time, temp, voltage = reconstruct(trace, processor)
        records.extend({"time_s": t, "gpu": processor, "temperature_c": y, "voltage_v": v}
                       for t, y, v in zip(time, temp, voltage))
        line = ax.plot(time, temp, color=colors[processor], label=f"GPU {processor} temperature")[0]
        limit = ax.axhline(60.0, color=THRESHOLD_RED, linestyle="--", linewidth=1.6,
                           label=r"$T_{max}=60\,^{\circ}$C")
        twin = ax.twinx()
        volts = twin.step(time, voltage, where="post", color=GREEN, alpha=0.55,
                          linewidth=1.25, label="Voltage")[0]
        twin.set_ylim(0.56, 0.92); twin.set_yticks(VOLTAGES)
        twin.set_ylabel("Voltage (V)")
        ax.set_ylabel(r"Temperature ($^{\circ}$C)")
        ax.grid(True, axis="both")
        over = [x for x in trace if x.get("event") == "complete" and x["processor"] == processor
                and x.get("overrun_factor", 1.0) > 1.0]
        for event in over:
            x = event["time"]
            y = event["temperature"]
            ax.scatter([x], [y], marker="*", s=120,
                       color="#CC0000", edgecolor="white", zorder=5)
            ax.annotate("hidden overrun", (x, y), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontsize=10.5, color="#9B0000")
        if processor == 1:
            inset = ax.inset_axes([0.52, 0.08, 0.43, 0.37])
            inset.plot(time, temp, color=colors[processor], linewidth=1.5)
            inset.axhline(60.0, color="#CC0000", linestyle="--", linewidth=1.1)
            for event in over:
                inset.scatter(event["time"], event["temperature"], marker="*", s=65,
                              color="#CC0000", edgecolor="white", zorder=5)
            inset.set_xlim(5.7, 8.35); inset.set_ylim(59.84, 60.08)
            inset.set_xticks([6, 7, 8]); inset.tick_params(labelsize=9.5)
            inset.grid(True)
        ax.legend([line, limit, volts], [line.get_label(), limit.get_label(), volts.get_label()],
                  loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=3, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.36)
    fig.savefig(OUT / "r2_6_hidden_overrun_trace.pdf")
    fig.savefig(OUT / "r2_6_hidden_overrun_trace.png")
    plt.close(fig)
    with (OUT / "r2_6_hidden_overrun_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    injected = [x for x in trace if x.get("event") == "dispatch" and x.get("overrun_factor", 1) > 1]
    (OUT / "trace_metadata.json").write_text(json.dumps({
        "summary": summary, "injected_dispatches": injected,
        "note": "Illustrative mechanism trace; formal statistics use random rare overruns."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
