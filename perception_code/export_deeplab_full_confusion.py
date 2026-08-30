from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from tqdm import tqdm

from train_csdnn_shared_corrected import MEAN, STD, prepare_frame, rle_to_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--split-protocol", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = args.project / "data" / "severstal-steel-defect-detection"
    frame = prepare_frame(data / "train.csv")
    ids = json.loads(args.split_protocol.read_text(encoding="utf-8"))["test_ids"]
    metric = pd.read_csv(args.metrics)
    threshold = np.asarray([float(metric[(metric["mode"] == "full") &
                                        (metric["class"].astype(str) == str(c))]
                                  .iloc[0].threshold) for c in range(1, 5)])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=4).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    rows = []
    with torch.inference_mode():
        for pool_index, image_id in enumerate(tqdm(ids, desc="export full-image confusion")):
            image = cv2.imread(str(data / "train_images" / image_id))
            value = (image.astype(np.float32) / 255.0 - MEAN) / STD
            x = torch.from_numpy(value.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            out = model(x); logits = out["out"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)[0].cpu().numpy()
            truth = np.stack([rle_to_mask(frame.loc[image_id, f"class_{c}"])
                              for c in range(1, 5)]).astype(bool)
            pred = prob >= threshold[:, None, None]
            tp = int(np.logical_and(pred, truth).sum())
            fp = int(np.logical_and(pred, ~truth).sum())
            fn = int(np.logical_and(~pred, truth).sum())
            rows.append({"pool_index": pool_index, "image_id": image_id,
                         "tp": tp, "fp": fp, "fn": fn, "gt": int(truth.sum())})
    pd.DataFrame(rows).to_csv(args.output / "full_image_confusion.csv", index=False)


if __name__ == "__main__":
    main()
