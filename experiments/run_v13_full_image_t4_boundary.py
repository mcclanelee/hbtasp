"""Full-image T4 p99.5 real-time boundary on the common periodic workload.

Each production line releases one indivisible full-image job per period.  The
two processor service budgets are the author-supplied 10,000-run T4 p99.5
budgets and the declared 0.5-performance second-slot profiles.  A stable
earliest-finish list scheduler admits a job only if it can finish by its
implicit deadline.  Rejected full images receive zero perception credit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import pandas as pd

from experiments.initial_manuscript_hbtasp import TAMB, VOLTAGES, end_temperature
from experiments.mandatory_terminal_registry import MandatoryTerminalRegistry
from experiments.run_v9_thermal_augmented_factorial import reconstruct_thermal


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json"
DEEPLAB_PATH = ROOT / "perception_evidence/deeplab_corrected_final/full_image_confusion.csv"
YOLO_PATH = ROOT / "perception_evidence/yolo_corrected_final_metrics/image_level_results.csv"
PROFILE_PATH = ROOT / "FULL_IMAGE_T4_PROFILE_LOCK_V1.json"
OUT = ROOT / "experiments/checkpoints/v13_full_image_t4_boundary"
RESULT = OUT / "cell_results.csv"
PERIODS = (100, 150, 200, 250, 300)
LINES = (4, 6, 8, 10)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EPOCHS = 1000
CONFIGS = ("DeepLabV3-MobileNetV3-Full", "YOLOv8n-Full")

_POOL = None
_DEEPLAB = None
_YOLO = None
_PROFILE = None


@dataclass(frozen=True)
class FullImageJob:
    uid: str
    image_uid: str
    release: float
    deadline: float
    source_image_id: str
    mandatory: bool = True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init_worker() -> None:
    global _POOL, _DEEPLAB, _YOLO, _PROFILE
    _POOL = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    with DEEPLAB_PATH.open(encoding="utf-8", newline="") as f:
        _DEEPLAB = {r["image_id"]: {k: int(r[k]) for k in ("tp", "fp", "fn", "gt")}
                    for r in csv.DictReader(f)}
    with YOLO_PATH.open(encoding="utf-8", newline="") as f:
        _YOLO = {r["image_id"]: {k: int(r[k]) for k in ("gt_boxes", "pred_boxes", "any_true_positive")}
                 for r in csv.DictReader(f)}
    _PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def build_releases(period_ms: int, lines: int, seed: int):
    rng = Random(seed)
    order = list(range(len(_POOL)))
    rng.shuffle(order)
    period = period_ms / 1000.0
    releases = []
    cursor = 0
    for epoch in range(EPOCHS):
        release = epoch * period
        batch = []
        for line in range(lines):
            item = _POOL[order[cursor % len(order)]]
            cursor += 1
            image_id = str(item["image_id"])
            uid = f"e{epoch}:l{line}:{image_id}:full"
            batch.append(FullImageJob(uid, uid, release, release + period, image_id))
        releases.append(batch)
    return releases


def run_scheduler(config: str, period_ms: int, lines: int, seed: int):
    releases = build_releases(period_ms, lines, seed)
    registry = MandatoryTerminalRegistry.from_release_batches(releases)
    key = "DeepLabV3-MobileNetV3-full-image" if config.startswith("DeepLab") else "YOLOv8n-full-image"
    model = _PROFILE["models"][key]
    durations = (model["gpu0_budget_ms"] / 1000.0,
                 model["gpu1_degraded_profile_ms"] / 1000.0)
    free = [0.0, 0.0]
    temperatures = [float(TAMB), float(TAMB)]
    temperature_times = [0.0, 0.0]
    trace = []
    rejected = 0
    completed = 0

    for batch in releases:
        for job in batch:
            choices = []
            for core in range(2):
                start = max(job.release, free[core])
                choices.append((start + durations[core], start, core))
            finish, start, core = min(choices)
            if finish > job.deadline + 1e-12:
                registry.transition(job.uid, "PRE_REJECT")
                rejected += 1
                trace.append({"event": "mandatory_infeasible", "time": job.release,
                              "uid": job.uid, "image_uid": job.image_uid,
                              "source_image_id": job.source_image_id,
                              "mandatory": True, "terminal_state": "PRE_REJECT"})
                continue
            if start > temperature_times[core]:
                temperatures[core] = end_temperature(
                    temperatures[core], VOLTAGES[0], start - temperature_times[core]
                )
                temperature_times[core] = start
            start_temp = temperatures[core]
            end_temp = end_temperature(start_temp, 0.8, durations[core])
            trace.append({"event": "dispatch", "time": start, "processor": core,
                          "uid": job.uid, "image_uid": job.image_uid,
                          "source_image_id": job.source_image_id, "mandatory": True,
                          "voltage": 0.8, "temperature": start_temp})
            registry.transition(job.uid, "ON_TIME")
            trace.append({"event": "complete", "time": finish, "processor": core,
                          "uid": job.uid, "image_uid": job.image_uid,
                          "source_image_id": job.source_image_id, "mandatory": True,
                          "terminal_state": "ON_TIME", "deadline_miss": False,
                          "voltage": 0.8, "temperature": end_temp})
            free[core] = finish
            temperatures[core] = end_temp
            temperature_times[core] = finish
            completed += 1
    summary = registry.finalize()
    summary.update({"released_images": lines * EPOCHS, "completed_images": completed,
                    "pre_rejected_images": rejected, "on_time_completion_rate": completed / (lines * EPOCHS)})
    return summary, trace


def score(config: str, trace: list[dict]):
    terminal = [e for e in trace if e["event"] in ("complete", "mandatory_infeasible")]
    on_time = {e["uid"] for e in terminal if e["event"] == "complete"}
    if config.startswith("DeepLab"):
        dice_sum = 0.0; tp = fp = fn = gt = 0; complete_miss = 0
        for e in terminal:
            c = _DEEPLAB[e["source_image_id"]]
            gt += c["gt"]
            if e["uid"] in on_time:
                tp += c["tp"]; fp += c["fp"]; fn += c["fn"]
                den = 2*c["tp"] + c["fp"] + c["fn"]
                dice_sum += 1.0 if den == 0 else 2*c["tp"] / den
                complete_miss += int(c["gt"] > 0 and c["tp"] == 0)
            else:
                fn += c["gt"]
                complete_miss += int(c["gt"] > 0)
        return {"deadline_aware_complete_image_dice": dice_sum / len(terminal),
                "deadline_aware_pixel_recall": tp / gt,
                "deadline_aware_image_complete_miss_rate": complete_miss / len(terminal),
                "native_metric_family": "segmentation"}
    detected = 0; defective = 0; misses = 0
    for e in terminal:
        y = _YOLO[e["source_image_id"]]
        if y["gt_boxes"] <= 0:
            continue
        defective += 1
        hit = e["uid"] in on_time and bool(y["any_true_positive"])
        detected += int(hit); misses += int(not hit)
    return {"deadline_aware_image_detection_recall": detected / defective,
            "deadline_aware_image_complete_miss_rate": misses / defective,
            "native_box_recall": 0.468, "native_map50": 0.449,
            "native_metric_family": "box_detection"}


def run_key(key):
    config, period, lines, seed = key
    summary, trace = run_scheduler(config, period, lines, seed)
    if summary["mandatory_terminal_residual"] != 0:
        raise RuntimeError(f"terminal residual: {key}")
    return {"configuration": config, "period_ms": period, "lines": lines,
            "seed": seed, "epochs": EPOCHS, **summary, **score(config, trace),
            **reconstruct_thermal(trace)}


def write_rows(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    tmp = RESULT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(RESULT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RESULT).to_dict("records") if RESULT.exists() else []
    done = {(r["configuration"], int(r["period_ms"]), int(r["lines"]), int(r["seed"])) for r in rows}
    keys = [(c,p,l,s) for c in CONFIGS for p in PERIODS for l in LINES for s in SEEDS]
    pending = [k for k in keys if k not in done]
    started = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as ex:
        futures = {ex.submit(run_key, k): k for k in pending}
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda r: (CONFIGS.index(r["configuration"]), int(r["period_ms"]), int(r["lines"]), int(r["seed"])))
            write_rows(rows)
            (OUT / "checkpoint.json").write_text(json.dumps({
                "status": "complete" if len(rows) == len(keys) else "running",
                "completed_cells": len(rows), "total_cells": len(keys),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds_this_run": time.time()-started,
                "profile_sha256": sha256(PROFILE_PATH), "pool_sha256": sha256(POOL_PATH),
                "protocol": {"periods_ms": PERIODS, "lines": LINES, "seeds": SEEDS,
                             "epochs": EPOCHS, "release": "periodic simultaneous per-line",
                             "scheduler": "stable earliest-finish admission",
                             "deadline_credit": "all-zero for rejected full image"}
            }, indent=2), encoding="utf-8")
            if len(rows) % 20 == 0: print(f"[{len(rows)}/{len(keys)}]", flush=True)
    if rows:
        checkpoint = json.loads((OUT / "checkpoint.json").read_text(encoding="utf-8"))
        checkpoint["status"] = "complete" if len(rows) == len(keys) else "running"
        checkpoint["sha256"] = {
            "profile": sha256(PROFILE_PATH), "pool": sha256(POOL_PATH),
            "deeplab_confusion": sha256(DEEPLAB_PATH), "yolo_image_results": sha256(YOLO_PATH),
            "runner": sha256(Path(__file__)), "results": sha256(RESULT),
        }
        (OUT / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
