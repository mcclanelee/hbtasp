from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from tqdm import tqdm

from train_csdnn_shared_corrected import (MEAN, STD, RegionDataset,
                                           prepare_frame, rle_to_mask,
                                           rle_touched_regions)


GRID = np.arange(.10, .951, .05, dtype=np.float32)


def loss_fn(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + target.sum((0, 2, 3))
    return bce + 1 - ((2 * inter + 1) / (denom + 1)).mean()


def forward(model, x):
    out = model(x)
    return out["out"] if isinstance(out, dict) else out


def migrate_legacy(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in current and current[k].shape == v.shape}
    # The old loader trained channels [Class2, Class3, Class4, empty].
    for prefix in ("classifier.4", "aux_classifier.4"):
        wk, bk = prefix + ".weight", prefix + ".bias"
        if wk in compatible:
            migrated_w = current[wk].clone()
            migrated_w[1:4] = compatible[wk][0:3]
            migrated_w[0].zero_()
            compatible[wk] = migrated_w
        if bk in compatible:
            migrated_b = current[bk].clone()
            migrated_b[1:4] = compatible[bk][0:3]
            migrated_b[0] = -4.0
            compatible[bk] = migrated_b
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "missing": len(missing),
            "unexpected": len(unexpected)}


def exact_pos_weight(frame, cap, device):
    positives = []
    for c in range(1, 5):
        total = 0
        for value in frame[f"class_{c}"].dropna():
            nums = list(map(int, str(value).split()))
            total += sum(nums[1::2])
        positives.append(total)
    pixels = len(frame) * 256 * 1600
    weights = [min(cap, (pixels - p) / max(1, p)) for p in positives]
    return positives, weights, torch.tensor(weights, device=device).view(1, 4, 1, 1)


def confusion(pred, truth):
    return np.asarray((np.logical_and(pred, truth).sum(),
                       np.logical_and(pred, ~truth).sum(),
                       np.logical_and(~pred, truth).sum()), np.int64)


def full_input(image):
    value = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(value.transpose(2, 0, 1)).float().unsqueeze(0)


def full_truth(row):
    return np.stack([rle_to_mask(row[f"class_{c}"]) for c in range(1, 5)]).astype(bool)


def collect(model, ids, frame, image_dir, device, mode, thresholds=None):
    shape = (len(GRID), 4, 3) if thresholds is None else (4, 3)
    counts = np.zeros(shape, np.int64)
    model.eval()
    with torch.inference_mode():
        for image_id in tqdm(ids, desc=f"deeplab {mode}"):
            image = cv2.imread(str(image_dir / image_id))
            truth = full_truth(frame.loc[image_id])
            if mode == "full":
                prob = torch.sigmoid(forward(model, full_input(image).to(device)))[0].cpu().numpy()
            else:
                pieces = []
                for region in range(4):
                    crop = image[:, region*400:(region+1)*400]
                    crop = cv2.copyMakeBorder(crop, 0, 0, 8, 8, cv2.BORDER_REFLECT_101)
                    piece = torch.sigmoid(forward(model, full_input(crop).to(device)))[0, :, :, 8:-8]
                    pieces.append(piece.cpu().numpy())
                prob = np.concatenate(pieces, axis=2)
            if thresholds is None:
                for ti, threshold in enumerate(GRID):
                    for c in range(4):
                        counts[ti, c] += confusion(prob[c] >= threshold, truth[c])
            else:
                for c in range(4):
                    counts[c] += confusion(prob[c] >= thresholds[c], truth[c])
    return counts


def calibrate_and_test(model, calibration_ids, test_ids, frame, image_dir, device, mode):
    calibration = collect(model, calibration_ids, frame, image_dir, device, mode)
    thresholds = np.zeros(4, np.float32)
    for c in range(4):
        scores = []
        for ti in range(len(GRID)):
            tp, fp, fn = calibration[ti, c]
            scores.append(2 * tp / max(1, 2 * tp + fp + fn))
        thresholds[c] = GRID[int(np.argmax(scores))]
    test = collect(model, test_ids, frame, image_dir, device, mode, thresholds)
    total = test.sum(0); tp, fp, fn = total
    rows = []
    for c in range(4):
        ctp, cfp, cfn = test[c]
        rows.append({"mode": mode, "class": c + 1, "threshold": float(thresholds[c]),
                     "tp": int(ctp), "fp": int(cfp), "fn": int(cfn),
                     "dice": 2 * ctp / max(1, 2 * ctp + cfp + cfn),
                     "recall": ctp / max(1, ctp + cfn)})
    rows.append({"mode": mode, "class": "micro", "threshold": None,
                 "tp": int(tp), "fp": int(fp), "fn": int(fn),
                 "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                 "recall": tp / max(1, tp + fn)})
    return rows


def benchmark(model, device, shape, warmup=50, runs=500):
    x = torch.randn(*shape, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            forward(model, x)
        torch.cuda.synchronize()
        values = []
        for _ in range(runs):
            start = time.perf_counter(); forward(model, x); torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    a = np.asarray(values)
    return {"shape": shape, "runs": runs, "mean_ms": float(a.mean()),
            "p99_ms": float(np.percentile(a, 99)),
            "p999_ms": float(np.percentile(a, 99.9)), "max_ms": float(a.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--split-protocol", type=Path, required=True)
    p.add_argument("--legacy-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--skip-evaluation", action="store_true")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(69); np.random.seed(69); torch.manual_seed(69)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = args.project / "data" / "severstal-steel-defect-detection"
    frame = prepare_frame(data / "train.csv")
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    train = frame.loc[split["train_ids"]]
    calibration_ids, test_ids = split["calibration_ids"], split["test_ids"]
    dataset = RegionDataset(train, data / "train_images", True, expand_regions=True)
    sample_weights = []
    for image_id in dataset.ids:
        row = train.loc[image_id]
        c1, c2 = rle_touched_regions(row["class_1"]), rle_touched_regions(row["class_2"])
        for region in range(4):
            sample_weights.append(max(4.0 if region in c1 else 1.0,
                                      40.0 if region in c2 else 1.0))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True)
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=4).to(device)
    migration = migrate_legacy(model, args.legacy_checkpoint, device)
    positives, weights, pos_weight = exact_pos_weight(train, 30.0, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch); model.train(); total = 0.0; count = 0
        for bi, (x, y, _, _) in enumerate(tqdm(loader, desc=f"deeplab epoch {epoch+1}")):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(forward(model, x), y, pos_weight)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss); count += 1
        history.append({"epoch": epoch + 1, "loss": total / max(1, count)})
        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1,
                    "label_protocol": "Classes 1-4; one-based RLE; source-ID 70/10/20",
                    "migration": migration}, args.output / "deeplab_corrected_latest.pth")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    protocol = {"migration": migration, "positive_pixels": positives,
                "positive_weights": weights, "device": str(device),
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.skip_evaluation:
        return
    rows = []
    for mode in ("full", "split"):
        rows += calibrate_and_test(model, calibration_ids, test_ids, frame,
                                   data / "train_images", device, mode)
    pd.DataFrame(rows).to_csv(args.output / "final_test_metrics.csv", index=False)
    timing = {"gpu": torch.cuda.get_device_name(0),
              "full": benchmark(model, device, (1, 3, 256, 1600)),
              "region_416": benchmark(model, device, (1, 3, 256, 416))}
    (args.output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
