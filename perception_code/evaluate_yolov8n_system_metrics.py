from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO


def read_gt(path, width=1600, height=256):
    rows = []
    if not path.exists():
        return np.zeros((0, 5), np.float32)
    for line in path.read_text(encoding="utf-8").splitlines():
        c, cx, cy, w, h = map(float, line.split())
        rows.append((int(c), (cx-w/2)*width, (cy-h/2)*height,
                     (cx+w/2)*width, (cy+h/2)*height))
    return np.asarray(rows, np.float32)


def any_match(pred, gt, threshold=.5):
    if len(pred) == 0 or len(gt) == 0:
        return False
    for p in pred:
        candidates = gt[gt[:, 0] == p[0]]
        for g in candidates:
            ix0, iy0 = max(p[1], g[1]), max(p[2], g[2])
            ix1, iy1 = min(p[3], g[3]), min(p[4], g[4])
            inter = max(0, ix1-ix0) * max(0, iy1-iy0)
            union = (p[3]-p[1])*(p[4]-p[2]) + (g[3]-g[1])*(g[4]-g[2]) - inter
            if inter / max(1e-9, union) >= threshold:
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--iou", type=float, default=.5)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    image_root = args.data / "images" / "all"; label_root = args.data / "labels" / "all"
    ids = [Path(x).name for x in (args.data / "test.txt").read_text(encoding="utf-8").splitlines()]
    rows, times = [], []
    for image_id in tqdm(ids, desc="YOLO final system metrics"):
        image = cv2.imread(str(image_root / image_id))
        start = time.perf_counter()
        result = model.predict(image, imgsz=640, conf=args.conf, iou=.7,
                               device=0, verbose=False)[0]
        torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
        if result.boxes is None or len(result.boxes) == 0:
            pred = np.zeros((0, 5), np.float32)
        else:
            cls = result.boxes.cls.cpu().numpy()[:, None]
            xyxy = result.boxes.xyxy.cpu().numpy()
            pred = np.concatenate((cls, xyxy), axis=1)
        gt = read_gt(label_root / image_id.replace(".jpg", ".txt"))
        rows.append({"image_id": image_id, "gt_boxes": len(gt), "pred_boxes": len(pred),
                     "any_true_positive": int(any_match(pred, gt, args.iou))})
    raw = pd.DataFrame(rows); raw.to_csv(args.output / "image_level_results.csv", index=False)
    a = np.asarray(times[20:])
    summary = {"confidence": args.conf, "matching_iou": args.iou,
               "test_images": len(raw), "images_with_any_matched_detection": int(raw.any_true_positive.sum()),
               "image_complete_miss_rate": float(1-raw.any_true_positive.mean()),
               "latency_protocol": "first 20 calls excluded; end-to-end predict including NMS",
               "mean_ms": float(a.mean()), "p99_ms": float(np.percentile(a, 99)),
               "p999_ms": float(np.percentile(a, 99.9)), "max_ms": float(a.max()),
               "box_metrics_ultralytics_final_test": {"precision": .516, "recall": .468,
                                                        "map50": .449, "map50_95": .203}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
