from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


PATHS = [(0, 0, 0, 0), (0, 1, 1, 0), (1, 1, 1, 1),
         (1, 2, 2, 1), (2, 2, 3, 2)]
MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)
STD = np.asarray((0.229, 0.224, 0.225), np.float32)


def load_architecture(source: Path):
    spec = importlib.util.spec_from_file_location("csdnn_corrected", source)
    module = importlib.util.module_from_spec(spec)
    text = source.read_text(encoding="utf-8")
    marker = "\nmodel_path = './models/model_unet_var"
    if marker in text:
        text = text[:text.index(marker)]
    exec(compile(text, str(source), "exec"), module.__dict__)
    return module


def rle_to_mask(value, h=256, w=1600):
    result = np.zeros(h * w, np.uint8)
    if pd.isna(value) or not str(value).strip():
        return result.reshape(h, w, order="F")
    numbers = list(map(int, str(value).split()))
    for one_based, length in zip(numbers[0::2], numbers[1::2]):
        start = one_based - 1
        result[start:start + length] = 1
    return result.reshape(h, w, order="F")


def prepare_frame(csv_path: Path):
    raw = pd.read_csv(csv_path)
    frame = raw.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    frame.columns = [f"class_{c}" for c in frame.columns]
    for c in range(1, 5):
        if f"class_{c}" not in frame:
            frame[f"class_{c}"] = np.nan
    frame["defects"] = frame[[f"class_{c}" for c in range(1, 5)]].notna().sum(axis=1)
    return frame


def rle_touched_regions(value, h=256):
    touched = set()
    if pd.isna(value) or not str(value).strip():
        return touched
    numbers = list(map(int, str(value).split()))
    for one_based, length in zip(numbers[0::2], numbers[1::2]):
        first_x = (one_based - 1) // h
        last_x = (one_based - 1 + length - 1) // h
        for region in range(max(0, first_x // 400), min(3, last_x // 400) + 1):
            touched.add(region)
    return touched


class RegionDataset(Dataset):
    def __init__(self, frame, image_dir: Path, train: bool, expand_regions=False):
        self.frame = frame
        self.ids = list(frame.index)
        self.image_dir = image_dir
        self.train = train
        self.expand_regions = expand_regions
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.ids) * (4 if self.expand_regions else 1)

    def __getitem__(self, index):
        image_index = index // 4 if self.expand_regions else index
        image_id = self.ids[image_index]
        row = self.frame.loc[image_id]
        image = cv2.imread(str(self.image_dir / image_id))
        if image is None:
            raise FileNotFoundError(image_id)
        # The corrected protocol covers all four 400-pixel regions every epoch.
        region = index % 4 if self.expand_regions else (index + self.epoch) % 4
        image = image[:, region * 400:(region + 1) * 400]
        mask = np.stack([rle_to_mask(row[f"class_{c}"])[:, region * 400:(region + 1) * 400]
                         for c in range(1, 5)], axis=0)
        if self.train:
            rng = random.random()
            if rng < .5:
                image = image[:, ::-1].copy(); mask = mask[:, :, ::-1].copy()
            if random.random() < .5:
                image = image[::-1].copy(); mask = mask[:, ::-1].copy()
        # Reflection padding retains every original pixel and makes width divisible by 32.
        image = cv2.copyMakeBorder(image, 0, 0, 8, 8, cv2.BORDER_REFLECT_101)
        mask = np.pad(mask, ((0, 0), (0, 0), (8, 8)), mode="constant")
        value = image.astype(np.float32) / 255.0
        value = (value - MEAN) / STD
        value = torch.from_numpy(value.transpose(2, 0, 1)).float()
        return value, torch.from_numpy(mask.astype(np.float32)), image_id, region


def loss_function(logits, target, positive_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target,
                                              pos_weight=positive_weight)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return bce + dice


def validate(model, dataset, device, max_images=0):
    model.eval()
    totals = {level: np.zeros(3, np.int64) for level in range(1, 6)}
    class_totals = {level: np.zeros((4, 3), np.int64) for level in range(1, 6)}
    ids = dataset.ids[:max_images or len(dataset.ids)]
    with torch.inference_mode():
        for image_id in tqdm(ids, desc="validate shared paths"):
            row = dataset.frame.loc[image_id]
            image = cv2.imread(str(dataset.image_dir / image_id))
            masks = []
            inputs = []
            for region in range(4):
                crop = image[:, region * 400:(region + 1) * 400]
                crop = cv2.copyMakeBorder(crop, 0, 0, 8, 8, cv2.BORDER_REFLECT_101)
                value = (crop.astype(np.float32) / 255.0 - MEAN) / STD
                inputs.append(torch.from_numpy(value.transpose(2, 0, 1)).float())
                mask = np.stack([rle_to_mask(row[f"class_{c}"])[:, region * 400:(region + 1) * 400]
                                 for c in range(1, 5)], axis=0)
                masks.append(np.pad(mask, ((0, 0), (0, 0), (8, 8)), mode="constant").astype(bool))
            batch = torch.stack(inputs).to(device)
            truth = np.stack(masks)
            for level, path in enumerate(PATHS, 1):
                estimate = (torch.sigmoid(model(batch, path)) > .5).cpu().numpy()
                for c in range(4):
                    tp = np.logical_and(estimate[:, c], truth[:, c]).sum()
                    fp = np.logical_and(estimate[:, c], ~truth[:, c]).sum()
                    fn = np.logical_and(~estimate[:, c], truth[:, c]).sum()
                    class_totals[level][c] += (tp, fp, fn)
                    totals[level] += (tp, fp, fn)
    rows = []
    for level in range(1, 6):
        tp, fp, fn = totals[level]
        rows.append({"level": level, "tp": int(tp), "fp": int(fp), "fn": int(fn),
                     "dice": float(2 * tp / max(1, 2 * tp + fp + fn)),
                     "recall": float(tp / max(1, tp + fn)),
                     "class_recall": [float(x[0] / max(1, x[0] + x[2]))
                                      for x in class_totals[level]]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=500)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--no-legacy-channel-remap", dest="legacy_channel_remap",
                        action="store_false", default=True,
                        help="disable migration of legacy outputs [Class2,Class3,Class4,empty]")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
    parser.add_argument("--class1-region-boost", type=float, default=8.0)
    parser.add_argument("--class2-region-boost", type=float, default=60.0)
    parser.add_argument("--path-weights", default="1,1,1,1,1",
                        help="comma-separated sampling weights for L1--L5")
    parser.add_argument("--train-fraction", type=float, default=.70)
    parser.add_argument("--calibration-fraction", type=float, default=.10)
    args = parser.parse_args()
    path_weights = [float(x) for x in args.path_weights.split(",")]
    if len(path_weights) != 5 or sum(path_weights) <= 0:
        raise ValueError("--path-weights must contain five nonnegative values")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    project = args.project.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    data = project / "data" / "severstal-steel-defect-detection"
    frame = prepare_frame(data / "train.csv")
    heldout_fraction = 1.0 - args.train_fraction
    train, heldout = train_test_split(frame, test_size=heldout_fraction,
                                      stratify=frame.defects, random_state=69)
    calibration_share = args.calibration_fraction / heldout_fraction
    heldout_strata = heldout.defects.clip(upper=2)
    val, final_test = train_test_split(heldout, train_size=calibration_share,
                                       stratify=heldout_strata, random_state=70)
    (args.output / "split_protocol.json").write_text(json.dumps({
        "seed_train": 69, "seed_calibration": 70,
        "train_ids": list(train.index), "calibration_ids": list(val.index),
        "test_ids": list(final_test.index)}, indent=2), encoding="utf-8")
    train_set = RegionDataset(train, data / "train_images", True, expand_regions=True)
    val_set = RegionDataset(val, data / "train_images", False)
    sample_weights = []
    for image_id in train_set.ids:
        row = train.loc[image_id]
        class1_regions = rle_touched_regions(row["class_1"])
        class2_regions = rle_touched_regions(row["class_2"])
        for region in range(4):
            weight = 1.0
            if region in class1_regions:
                weight = max(weight, args.class1_region_boost)
            if region in class2_regions:
                weight = max(weight, args.class2_region_boost)
            sample_weights.append(weight)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True)
    architecture = load_architecture(project / "code" / "model_unet_var.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = architecture.UNet().to(device)
    init = args.init or project / "models" / "model_unet_var_level5_best.pth"
    if not args.scratch:
        checkpoint = torch.load(init, map_location=device, weights_only=False)
        state = checkpoint["state_dict"]
        legacy_checkpoint = "label_protocol" not in checkpoint
        if args.legacy_channel_remap and legacy_checkpoint:
            # The legacy loader trained output channels as Class2, Class3,
            # Class4, and an empty scalar-derived channel.  Preserve those
            # useful filters in their corrected locations and initialize the
            # previously unseen Class1 output from the model default.
            current = model.state_dict()
            for key in ("decoder.final_conv.weight", "decoder.final_conv.bias"):
                migrated = current[key].clone()
                migrated[1:4] = state[key][0:3]
                state[key] = migrated
            # Class 1 was absent from legacy training.  A negative prior avoids
            # treating an untrained random head as foreground everywhere.
            state["decoder.final_conv.weight"][0].zero_()
            state["decoder.final_conv.bias"][0] = -4.0
        model.load_state_dict(state)
        del checkpoint, state
    # Exact foreground counts are available directly from RLE lengths.  Cap
    # the weights to keep rare classes from destabilising shared-path updates.
    positives = []
    for c in range(1, 5):
        total = 0
        for value in train[f"class_{c}"].dropna():
            numbers = list(map(int, str(value).split()))
            total += sum(numbers[1::2])
        positives.append(total)
    pixels = len(train) * 256 * 1600
    weights = [min(args.max_pos_weight, (pixels - p) / max(1, p)) for p in positives]
    positive_weight = torch.tensor(weights, device=device).view(1, 4, 1, 1)
    (args.output / "class_balance.json").write_text(
        json.dumps({"positive_pixels": positives, "positive_weight": weights}, indent=2),
        encoding="utf-8")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        start = time.time(); running = 0.0; count = 0
        for batch_index, (images, masks, _, _) in enumerate(tqdm(loader, desc=f"epoch {epoch+1}")):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            level = random.choices(range(5), weights=path_weights, k=1)[0]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images, PATHS[level])
                loss = loss_function(logits, masks, positive_weight) / args.accumulation
            scaler.scale(loss).backward()
            if (batch_index + 1) % args.accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * args.accumulation; count += 1
        metrics = validate(model, val_set, device, args.max_val_images)
        record = {"epoch": epoch + 1, "loss": running / max(1, count),
                  "seconds": time.time() - start, "paths": metrics}
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "metrics": metrics,
                    "label_protocol": "Classes 1-4; RLE one-based; 400->416 reflect pad",
                    "path_protocol": "single shared checkpoint, sampled paths",
                    "path_weights": path_weights},
                   args.output / "csdnn_shared_corrected_latest.pth")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
