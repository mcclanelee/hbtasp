"""Priority-discrimination envelope on the frozen principal grid.

This replay keeps HBTASP, the calibrated test pool, the mask-confusion evidence,
and the 100--300 ms scheduling protocol unchanged.  It compares the deployed
calibrated-histogram ranking with a frozen label-free random ranking and a
  analysis-only ground-truth-informed diagnostic reference. The existing,
  validated histogram cells from v21 are reused; only the two comparison
  envelopes are replayed.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.initial_manuscript_event_replay import run_continuous_hbtasp
from experiments.initial_r2_8_metrics import (
    load_confusion,
    score_historical_scalar,
    score_trace,
)


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
CONFUSION_PATH = ROOT / "mask_replay_final_test_shared/mask_confusion_by_level.csv"
V21_RESULT = ROOT / "experiments/checkpoints/v21_multicue_priority_aligned/cell_results.csv"
OUT = ROOT / "experiments/checkpoints/v22_priority_discrimination_envelope"
RESULT = OUT / "cell_results.csv"

STRATEGIES = ("calibrated_histogram", "random_frozen", "defect_oracle")
NEW_STRATEGIES = ("random_frozen", "defect_oracle")
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EPOCHS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_random_weights(image_id: str) -> list[float]:
    values = []
    for region in range(4):
        token = f"review2-priority-envelope:{image_id}:{region}".encode("utf-8")
        values.append(int(hashlib.sha256(token).hexdigest()[:16], 16) + 1)
    total = float(sum(values))
    return [value / total for value in values]


def transform_pool(source: list[dict], strategy: str) -> list[dict]:
    result = []
    for source_item in source:
        item = copy.deepcopy(source_item)
        pixels = [int(value) for value in item["sub_pixels"]]
        if strategy == "calibrated_histogram":
            pass
        elif strategy == "random_frozen":
            item["weights"] = frozen_random_weights(str(item["image_id"]))
            item["mandatory_idx"] = int(max(range(4), key=lambda i: item["weights"][i]))
            item["priority_protocol"] = "frozen label-free SHA256 random ranking"
        elif strategy == "defect_oracle":
            total = sum(pixels)
            item["weights"] = [value / total for value in pixels] if total else [0.25] * 4
            item["mandatory_idx"] = int(max(range(4), key=lambda i: pixels[i]))
            item["priority_protocol"] = "analysis-only ground-truth-informed ranking"
        else:
            raise ValueError(strategy)
        result.append(item)
    return result


def priority_diagnostics(pool: list[dict]) -> dict[str, float]:
    defective = [item for item in pool if sum(item["sub_pixels"]) > 0]
    hits, shares = [], []
    for item in defective:
        chosen = int(item["mandatory_idx"])
        pixels = [int(value) for value in item["sub_pixels"]]
        hits.append(int(pixels[chosen] > 0))
        shares.append(pixels[chosen] / sum(pixels))
    return {
        "priority_defect_hit_rate": sum(hits) / len(hits),
        "priority_defect_pixel_share": sum(shares) / len(shares),
    }


def mandatory_stats(trace: list[dict]) -> tuple[int, int, int]:
    terminal = {"complete", "mandatory_infeasible", "optional_skipped", "expired", "dispatch_infeasible"}
    rows = [event for event in trace if event.get("event") in terminal and event.get("mandatory")]
    misses = sum(event["event"] != "complete" or event.get("deadline_miss", False) for event in rows)
    admitted_late = sum(event["event"] == "complete" and event.get("deadline_miss", False) for event in rows)
    return len(rows), misses, admitted_late


def write_rows(rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = RESULT.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULT)


def load_or_seed_rows(diagnostics: dict[str, dict[str, float]]) -> list[dict]:
    if RESULT.exists():
        with RESULT.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    # The published v21 CSV carries a UTF-8 BOM before its quoted header.
    with V21_RESULT.open(newline="", encoding="utf-8-sig") as stream:
        rows = []
        for source in csv.DictReader(stream):
            if source["treatment"] != "histogram_zero":
                continue
            row = dict(source)
            row.pop("treatment")
            row["strategy"] = "calibrated_histogram"
            row.update(diagnostics["calibrated_histogram"])
            row["evidence_origin"] = "reused validated v21 histogram_zero cell"
            rows.append(row)
    write_rows(rows)
    return rows


def validate_histogram_seed(rows: list[dict]) -> None:
    """Confirm that the v22 histogram cells equal the current final v21 cells."""
    with V21_RESULT.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = [row for row in csv.DictReader(stream)
                       if row["treatment"] == "histogram_zero"]
    target_rows = [row for row in rows if row["strategy"] == "calibrated_histogram"]
    key = lambda row: (int(row["period_ms"]), int(row["lines"]), int(row["seed"]))
    source = {key(row): row for row in source_rows}
    target = {key(row): row for row in target_rows}
    if len(source) != 200 or source.keys() != target.keys():
        raise RuntimeError("v21/v22 calibrated-histogram grids do not match")
    fields = [field for field in source_rows[0] if field != "treatment"]
    for cell in source:
        if any(source[cell][field] != target[cell].get(field) for field in fields):
            raise RuntimeError(f"v21/v22 calibrated-histogram mismatch at {cell}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    pools = {strategy: transform_pool(source, strategy) for strategy in STRATEGIES}
    diagnostics = {strategy: priority_diagnostics(pool) for strategy, pool in pools.items()}
    (OUT / "priority_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    confusion = load_confusion(CONFUSION_PATH)

    rows = load_or_seed_rows(diagnostics)
    validate_histogram_seed(rows)
    done = {(row["strategy"], int(row["period_ms"]), int(row["lines"]), int(row["seed"]))
            for row in rows}
    total = len(STRATEGIES) * len(PERIODS) * len(LINES) * len(SEEDS)
    started = time.time()
    for strategy in NEW_STRATEGIES:
        pool = pools[strategy]
        for period in PERIODS:
            for lines in LINES:
                for seed in SEEDS:
                    key = (strategy, period, lines, seed)
                    if key in done:
                        continue
                    cell_started = time.time()
                    summary, trace = run_continuous_hbtasp(
                        pool, period, lines, EPOCHS, seed,
                        budget_mode="assigned_level_sensitivity",
                        network_mode="dynamic",
                        priority_overhead_ms=0.0,
                    )
                    total_mandatory, missed_mandatory, admitted_late = mandatory_stats(trace)
                    scalar = score_historical_scalar(trace, summary["total_regions"])
                    mask, _ = score_trace(trace, pool, confusion)
                    rows.append({
                        "strategy": strategy,
                        **summary,
                        "total_mandatory": total_mandatory,
                        "missed_mandatory": missed_mandatory,
                        "mandatory_service_failure": missed_mandatory / total_mandatory,
                        "admitted_mandatory_deadline_violations": admitted_late,
                        **diagnostics[strategy],
                        **scalar,
                        **mask,
                        "runtime_seconds": time.time() - cell_started,
                        "evidence_origin": "new v22 replay",
                    })
                    write_rows(rows)
                    done.add(key)
                    checkpoint = {
                        "status": "complete" if len(done) == total else "running",
                        "completed_cells": len(done),
                        "total_cells": total,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds_this_run": time.time() - started,
                        "protocol": {
                            "strategies": STRATEGIES,
                            "periods_ms": PERIODS,
                            "lines": LINES,
                            "seeds": SEEDS,
                            "epochs": EPOCHS,
                            "priority_overhead_ms": 0.0,
                            "histogram_cells_reused_from": str(V21_RESULT.relative_to(ROOT)),
                            "test_labels_used_by_calibrated_histogram": False,
                            "oracle_is_deployable": False,
                        },
                        "sha256": {
                            "calibrated_pool": sha256(POOL_PATH),
                            "confusion": sha256(CONFUSION_PATH),
                            "v21_source_results": sha256(V21_RESULT),
                            "results": sha256(RESULT),
                        },
                    }
                    (OUT / "checkpoint.json").write_text(
                        json.dumps(checkpoint, indent=2), encoding="utf-8")
                    print(f"[{len(done)}/{total}] {strategy} T={period} lines={lines} seed={seed}",
                          flush=True)

    checkpoint = {
        "status": "complete" if len(done) == total else "running",
        "completed_cells": len(done),
        "total_cells": total,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_this_run": time.time() - started,
        "protocol": {
            "strategies": STRATEGIES,
            "periods_ms": PERIODS,
            "lines": LINES,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "priority_overhead_ms": 0.0,
            "histogram_cells_reused_from": str(V21_RESULT.relative_to(ROOT)),
            "histogram_subset_equivalence_verified": True,
            "test_labels_used_by_calibrated_histogram": False,
            "oracle_is_deployable": False,
        },
        "sha256": {
            "calibrated_pool": sha256(POOL_PATH),
            "confusion": sha256(CONFUSION_PATH),
            "v21_source_results": sha256(V21_RESULT),
            "results": sha256(RESULT),
        },
    }
    (OUT / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
