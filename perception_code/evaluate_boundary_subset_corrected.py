"""Dedicated evaluation of defects crossing the original 400-pixel cuts.

The subset is defined from ground truth before predictions are inspected.  A
class-specific connected component is a boundary defect when it contains pixels
on both sides of x=400, 800, or 1200.  Pixel Dice is evaluated in a 32-pixel
dilated component bounding box so false positives near the defect are counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from evaluate_overlap_corrected import PARTITIONS, predict_partition
from train_csdnn_shared_corrected import prepare_frame
from train_evaluate_deeplab_corrected import confusion, full_truth

BOUNDARIES = (400, 800, 1200)
ROI_MARGIN = 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing_components(truth: np.ndarray) -> list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]]:
    result = []
    for class_index in range(truth.shape[0]):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            truth[class_index].astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            x, y, width, height, _area = stats[label]
            right = x + width
            if not any(x < boundary < right for boundary in BOUNDARIES):
                continue
            component = labels == label
            x0 = max(0, x - ROI_MARGIN); x1 = min(1600, right + ROI_MARGIN)
            y0 = max(0, y - ROI_MARGIN); y1 = min(256, y + height + ROI_MARGIN)
            result.append((class_index, label, component, (x0, y0, x1, y1)))
    return result


def evaluate_arm(model, ids, frame, image_dir, crops, thresholds, device):
    total_confusion = np.zeros(3, np.int64)
    class_confusion = np.zeros((4, 3), np.int64)
    class_components = np.zeros(4, np.int64)
    class_any_hit = np.zeros(4, np.int64)
    class_iou10 = np.zeros(4, np.int64)
    component_count = any_hit = iou10 = 0
    boundary_images: set[str] = set()
    complete_miss_images: set[str] = set()
    component_rows = []
    image_rows = []
    with torch.inference_mode():
        for image_id in ids:
            truth = full_truth(frame.loc[image_id])
            components = crossing_components(truth)
            if not components:
                continue
            boundary_images.add(image_id)
            image = cv2.imread(str(image_dir / image_id))
            prob = predict_partition(model, image, crops, device)
            prediction = prob >= thresholds[:, None, None]
            image_hit = False
            for class_index, component_label, component, (x0, y0, x1, y1) in components:
                target_roi = component[y0:y1, x0:x1]
                pred_roi = prediction[class_index, y0:y1, x0:x1]
                values = confusion(pred_roi, target_roi)
                total_confusion += values
                class_confusion[class_index] += values
                intersection = np.logical_and(prediction[class_index], component).sum()
                union = np.logical_or(pred_roi, target_roi).sum()
                hit = intersection > 0
                pass_iou = intersection / max(1, union) >= 0.10
                component_count += 1
                any_hit += int(hit); iou10 += int(pass_iou)
                class_components[class_index] += 1
                class_any_hit[class_index] += int(hit)
                class_iou10[class_index] += int(pass_iou)
                image_hit = image_hit or hit
                ctp, cfp, cfn = values
                component_rows.append({
                    "image_id": image_id, "class_index": class_index,
                    "component_label": component_label,
                    "tp": int(ctp), "fp": int(cfp), "fn": int(cfn),
                    "roi_dice": float(2 * ctp / max(1, 2 * ctp + cfp + cfn)),
                    "pixel_recall": float(ctp / max(1, ctp + cfn)),
                    "any_overlap": int(hit), "iou10": int(pass_iou),
                })
            if not image_hit:
                complete_miss_images.add(image_id)
            image_rows.append({"image_id": image_id, "any_boundary_hit": int(image_hit),
                               "complete_miss": int(not image_hit)})
    tp, fp, fn = total_confusion
    class_dice = 2 * class_confusion[:, 0] / np.maximum(
        1, 2 * class_confusion[:, 0] + class_confusion[:, 1] + class_confusion[:, 2])
    class_recall = class_confusion[:, 0] / np.maximum(
        1, class_confusion[:, 0] + class_confusion[:, 2])
    summary = {
        "boundary_images": len(boundary_images),
        "crossing_components": int(component_count),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "boundary_component_roi_dice": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "boundary_component_pixel_recall": float(tp / max(1, tp + fn)),
        "component_any_overlap_recall": float(any_hit / max(1, component_count)),
        "component_iou10_recall": float(iou10 / max(1, component_count)),
        "boundary_image_complete_miss_rate": float(
            len(complete_miss_images) / max(1, len(boundary_images))),
        "boundary_image_complete_misses": len(complete_miss_images),
        "class_component_counts": class_components.tolist(),
        "class_roi_dice": class_dice.tolist(),
        "class_pixel_recall": class_recall.tolist(),
        "class_any_overlap_recall": (class_any_hit / np.maximum(1, class_components)).tolist(),
        "class_iou10_recall": (class_iou10 / np.maximum(1, class_components)).tolist(),
    }
    return summary, component_rows, image_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overlap-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = args.project / "data" / "severstal-steel-defect-detection"
    frame = prepare_frame(data_dir / "train.csv")
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    overlap = json.loads(args.overlap_details.read_text(encoding="utf-8"))
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=4).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()

    details = {
        "status": "running", "definition": "ground-truth component crosses x=400/800/1200",
        "roi_margin_px": ROI_MARGIN, "boundaries_px": BOUNDARIES,
        "device": str(device), "measurement_hardware": "T4_measurement",
        "split_protocol": str(args.split_protocol), "checkpoint": str(args.checkpoint),
        "sha256": {"split": sha256(args.split_protocol),
                   "checkpoint": sha256(args.checkpoint),
                   "overlap_details": sha256(args.overlap_details)},
    }
    rows = []
    for name, crops in PARTITIONS.items():
        thresholds = np.asarray(overlap[name]["thresholds"], np.float32)
        result, component_rows, image_rows = evaluate_arm(
            model, split["test_ids"], frame, data_dir / "train_images",
            crops, thresholds, device)
        details[name] = result
        rows.append({"partition": name, **{k: v for k, v in result.items()
                    if not isinstance(v, list)}})
        pd.DataFrame(rows).to_csv(args.output / "boundary_subset_summary.csv", index=False)
        pd.DataFrame(component_rows).assign(partition=name).to_csv(
            args.output / f"{name}_components.csv", index=False)
        pd.DataFrame(image_rows).assign(partition=name).to_csv(
            args.output / f"{name}_images.csv", index=False)
        (args.output / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(name, result, flush=True)
    details["status"] = "complete"
    (args.output / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
