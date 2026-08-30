"""End-to-end R2.8 scoring; every non-on-time defective region is FN."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from experiments.initial_manuscript_hbtasp import DICE


def load_confusion(path: Path) -> dict[tuple[str, int, int], dict[str, int]]:
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result[(row["image_id"], int(row["sub_index"]), int(row["level"]))] = {
                key: int(row[key]) for key in ("tp", "fp", "fn", "gt")
            }
    return result


def score_trace(trace: Sequence[dict], pool: Sequence[dict], confusion: dict):
    pool_by_id = {item["image_id"]: item for item in pool}
    terminal = {"complete", "mandatory_infeasible", "optional_skipped",
                "expired", "dispatch_infeasible"}
    images = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0,
        "completed_tp": 0, "completed_fp": 0, "completed_fn": 0,
        "regions": 0, "on_time_regions": 0})
    seen = set()
    for event in trace:
        if event.get("event") not in terminal:
            continue
        if event["uid"] in seen:
            raise ValueError(f"duplicate terminal event: {event['uid']}")
        seen.add(event["uid"])
        source = event["source_image_id"]
        region = int(event["region_index"])
        gt = int(pool_by_id[source]["sub_pixels"][region])
        target = images[event["image_uid"]]
        target["gt"] += gt
        target["regions"] += 1
        on_time = event["event"] == "complete" and not event.get("deadline_miss", False)
        if on_time:
            values = confusion[(source, region, int(event["level"]))]
            for key in ("tp", "fp", "fn"):
                target[key] += values[key]
                target[f"completed_{key}"] += values[key]
            target["on_time_regions"] += 1
        else:
            target["fn"] += gt

    rows = []
    for image_uid, values in sorted(images.items()):
        den = 2 * values["tp"] + values["fp"] + values["fn"]
        cden = 2 * values["completed_tp"] + values["completed_fp"] + values["completed_fn"]
        dice = 1.0 if den == 0 else 2 * values["tp"] / den
        completed_dice = None if cden == 0 else 2 * values["completed_tp"] / cden
        rows.append({"image_uid": image_uid, **values,
            "complete_image_dice": dice, "completed_only_dice": completed_dice,
            "defect_recall": 1.0 if values["gt"] == 0 else values["tp"] / values["gt"],
            "complete_miss": int(values["gt"] > 0 and values["tp"] == 0)})
    if not rows:
        raise ValueError("trace has no terminal regions")
    defective = [row for row in rows if row["gt"] > 0]
    total_gt = sum(row["gt"] for row in rows)
    total_tp = sum(row["tp"] for row in rows)
    completed_rows = [x for x in rows if x["completed_only_dice"] is not None]
    summary = {"images": len(rows), "defective_images": len(defective),
        "mean_complete_image_dice": sum(x["complete_image_dice"] for x in rows) / len(rows),
        "completed_only_images": len(completed_rows),
        "mean_completed_only_dice": (
            sum(x["completed_only_dice"] for x in completed_rows) / len(completed_rows)
            if completed_rows else float("nan")
        ),
        "pixel_defect_recall": total_tp / total_gt,
        "pixel_missed_detection_rate": 1.0 - total_tp / total_gt,
        "image_complete_miss_rate": sum(x["complete_miss"] for x in defective) / len(defective)}
    summary["optimism_gap"] = (summary["mean_completed_only_dice"]
                               - summary["mean_complete_image_dice"])
    return summary, rows


def score_historical_scalar(trace: Sequence[dict], total_regions: int) -> dict:
    terminal = {"complete", "mandatory_infeasible", "optional_skipped",
                "expired", "dispatch_infeasible"}
    records = [event for event in trace if event.get("event") in terminal]
    if len(records) != total_regions:
        raise ValueError("scalar score requires one terminal event per region")
    on_time = [event for event in records if event["event"] == "complete"
               and not event.get("deadline_miss", False)]
    quality_sum = sum(DICE[int(event["level"]) - 1] for event in on_time)
    mandatory = [event for event in records if event.get("mandatory")]
    mandatory_on_time = [event for event in mandatory if event["event"] == "complete"
                         and not event.get("deadline_miss", False)]
    total_weight = sum(float(event.get("weight", 1.0)) for event in records)
    weighted_quality = sum(float(event.get("weight", 1.0))
                           * DICE[int(event["level"]) - 1] for event in on_time)
    return {
        "historical_completed_only_dice": quality_sum / len(on_time) if on_time else 0.0,
        "historical_coverage_adjusted_dice": quality_sum / total_regions,
        "historical_mandatory_effective_dice": (
            sum(DICE[int(event["level"]) - 1] for event in mandatory_on_time)
            / len(mandatory) if mandatory else 0.0
        ),
        "on_time_completed_regions": len(on_time),
        "historical_weighted_coverage_utility": weighted_quality / total_weight,
    }
