from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from tqdm import tqdm

from train_csdnn_shared_corrected import MEAN, STD, prepare_frame, rle_to_mask
from train_evaluate_deeplab_corrected import GRID, confusion, forward, full_input, full_truth


PARTITIONS = {
    "non_overlap_4x400": [(0, 400), (400, 400), (800, 400), (1200, 400)],
    "mild_overlap_4x416": [(0, 416), (395, 416), (790, 416), (1184, 416)],
    "heavy_overlap_5x400": [(0, 400), (300, 400), (600, 400), (900, 400), (1200, 400)],
}


def crop_input(image: np.ndarray, start: int, width: int) -> tuple[torch.Tensor, int, int]:
    end = min(1600, start + width)
    crop = image[:, start:end]
    actual = end - start
    # The final CS-DNN/DeepLab regional protocol uses 416-pixel inputs and
    # reflection padding. Keep that contract for every 400-pixel crop.
    if actual < 416:
        total = 416 - actual
        left = total // 2
        right = total - left
        crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_REFLECT_101)
    else:
        left = 0
    return full_input(crop), left, actual


def predict_partition(model, image: np.ndarray, crops, device) -> np.ndarray:
    summed = np.zeros((4, 256, 1600), np.float32)
    weight = np.zeros((256, 1600), np.float32)
    for start, width in crops:
        x, left, actual = crop_input(image, start, width)
        prob = torch.sigmoid(forward(model, x.to(device)))[0].cpu().numpy()
        prob = prob[:, :, left:left + actual]
        end = start + actual
        summed[:, :, start:end] += prob
        weight[:, start:end] += 1
    if np.any(weight == 0):
        raise RuntimeError("partition leaves uncovered pixels")
    return summed / weight[None]


def calibrate(model, ids, frame, image_dir, crops, device):
    counts = np.zeros((len(GRID), 4, 3), np.int64)
    with torch.inference_mode():
        for image_id in tqdm(ids, desc="calibrate overlap", disable=True):
            image = cv2.imread(str(image_dir / image_id))
            truth = full_truth(frame.loc[image_id])
            prob = predict_partition(model, image, crops, device)
            for ti, threshold in enumerate(GRID):
                for c in range(4):
                    counts[ti, c] += confusion(prob[c] >= threshold, truth[c])
    thresholds = []
    for c in range(4):
        dice = [2 * v[0] / max(1, 2 * v[0] + v[1] + v[2]) for v in counts[:, c]]
        thresholds.append(float(GRID[int(np.argmax(dice))]))
    return np.asarray(thresholds, np.float32)


def evaluate(model, ids, frame, image_dir, crops, thresholds, device):
    per_class = np.zeros((4, 3), np.int64)
    defect_images = complete_misses = 0
    with torch.inference_mode():
        for image_id in tqdm(ids, desc="test overlap", disable=True):
            image = cv2.imread(str(image_dir / image_id))
            truth = full_truth(frame.loc[image_id])
            prob = predict_partition(model, image, crops, device)
            prediction = prob >= thresholds[:, None, None]
            for c in range(4):
                per_class[c] += confusion(prediction[c], truth[c])
            if truth.any():
                defect_images += 1
                if not np.logical_and(prediction, truth).any():
                    complete_misses += 1
    tp, fp, fn = per_class.sum(0)
    class_dice = 2 * per_class[:, 0] / np.maximum(
        1, 2 * per_class[:, 0] + per_class[:, 1] + per_class[:, 2])
    class_recall = per_class[:, 0] / np.maximum(1, per_class[:, 0] + per_class[:, 2])
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "micro_dice": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "micro_recall": float(tp / max(1, tp + fn)),
        "macro_dice": float(class_dice.mean()),
        "macro_recall": float(class_recall.mean()),
        "image_complete_miss_rate": float(complete_misses / max(1, defect_images)),
        "defect_images": defect_images, "complete_misses": complete_misses,
        "class_dice": class_dice.tolist(), "class_recall": class_recall.tolist(),
    }


def benchmark(model, crops, device, warmup=50, runs=300):
    image = np.random.default_rng(69).integers(0, 256, (256, 1600, 3), np.uint8)
    with torch.inference_mode():
        for _ in range(warmup):
            predict_partition(model, image, crops, device)
        torch.cuda.synchronize()
        values = []
        for _ in range(runs):
            start = time.perf_counter()
            predict_partition(model, image, crops, device)
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    a = np.asarray(values)
    return {"runs": runs, "mean_ms": float(a.mean()),
            "p99_ms": float(np.percentile(a, 99)),
            "p999_ms": float(np.percentile(a, 99.9)), "max_ms": float(a.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--split-protocol", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = args.project / "data" / "severstal-steel-defect-detection"
    frame = prepare_frame(data / "train.csv")
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=4).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    rows = []
    details = {"device": str(device), "split_protocol": str(args.split_protocol),
               "checkpoint": str(args.checkpoint), "partitions": PARTITIONS}
    for name, crops in PARTITIONS.items():
        thresholds = calibrate(model, split["calibration_ids"], frame,
                               data / "train_images", crops, device)
        result = evaluate(model, split["test_ids"], frame,
                          data / "train_images", crops, thresholds, device)
        timing = benchmark(model, crops, device)
        details[name] = {"thresholds": thresholds.tolist(), "metrics": result,
                         "timing": timing}
        rows.append({"partition": name, "n_crops": len(crops),
                     **{k: result[k] for k in ("micro_dice", "micro_recall",
                                               "macro_dice", "macro_recall",
                                               "image_complete_miss_rate")},
                     "mean_ms": timing["mean_ms"], "p99_ms": timing["p99_ms"]})
        # Persist each completed arm so an interrupted long GPU run remains
        # auditable and can be resumed manually without relying on console logs.
        pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
        (args.output / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
