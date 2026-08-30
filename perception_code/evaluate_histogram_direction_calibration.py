from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from evaluate_histogram_priority_corrected import histogram, perturb
from train_csdnn_shared_corrected import prepare_frame, rle_to_mask


MODES = ["original", "bright+20", "bright-20", "contrast+20%", "contrast-20%"]


def manuscript_chi_square(value, reference, eps=1e-9):
    """Equation (chi_square) in the initial manuscript: background denominator."""
    return float(np.sum((value - reference) ** 2 / (reference + eps)))


def build_templates(frame, image_dir, maximum):
    pools = [[] for _ in range(4)]
    for image_id, row in tqdm(frame.iterrows(), total=len(frame), desc="template train"):
        image = cv2.imread(str(image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        combined = np.logical_or.reduce([rle_to_mask(row[f"class_{c}"]) for c in range(1, 5)])
        for region in range(4):
            if len(pools[region]) >= maximum:
                continue
            x0, x1 = region * 400, (region + 1) * 400
            if not combined[:, x0:x1].any():
                pools[region].append(histogram(image[:, x0:x1]))
        if all(len(pool) >= maximum for pool in pools):
            break
    return [np.mean(pool, axis=0) for pool in pools], [len(pool) for pool in pools]


def score_partition(frame, image_dir, references, modes, label):
    rows = []
    for image_id, row in tqdm(frame.iterrows(), total=len(frame), desc=label):
        image = cv2.imread(str(image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        combined = np.logical_or.reduce([rle_to_mask(row[f"class_{c}"]) for c in range(1, 5)])
        for region in range(4):
            x0, x1 = region * 400, (region + 1) * 400
            pixels = int(combined[:, x0:x1].sum())
            crop = image[:, x0:x1]
            for mode in modes:
                distance = manuscript_chi_square(
                    histogram(perturb(crop, mode)), references[region])
                rows.append({"image_id": image_id, "region": region, "mode": mode,
                             "defect_pixels": pixels, "defective": int(pixels > 0),
                             "distance": distance})
    return pd.DataFrame(rows)


def summarize(frame, direction):
    records = []
    for mode, cell in frame.groupby("mode"):
        score = direction * cell.distance.to_numpy(float)
        y = cell.defective.to_numpy(int)
        rho, rho_p = spearmanr(score, cell.defect_pixels.to_numpy(float))
        eligible = cell.groupby("image_id").defective.max()
        ids = eligible[eligible > 0].index
        ranked = cell[cell.image_id.isin(ids)].copy()
        ranked["calibrated_score"] = direction * ranked.distance
        top = ranked.sort_values(["image_id", "calibrated_score"], ascending=[True, False]).groupby("image_id").first()
        records.append({"mode": mode, "regions": len(cell), "defective_regions": int(y.sum()),
                        "roc_auc": roc_auc_score(y, score),
                        "pr_auc": average_precision_score(y, score),
                        "spearman_area": rho, "spearman_p": rho_p,
                        "top1_defective_region_hit_rate": top.defective.mean()})
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--split-protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-template-regions", type=int, default=2000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.project / "data" / "severstal-steel-defect-detection"
    image_dir = data / "train_images"
    all_rows = prepare_frame(data / "train.csv")
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    train = all_rows.loc[split["train_ids"]]
    validation = all_rows.loc[split["calibration_ids"]]
    test = all_rows.loc[split["test_ids"]]
    refs, counts = build_templates(train, image_dir, args.max_template_regions)
    val_raw = score_partition(validation, image_dir, refs, ["original"], "validation calibration")
    auc_positive = roc_auc_score(val_raw.defective, val_raw.distance)
    direction = 1 if auc_positive >= .5 else -1
    test_raw = score_partition(test, image_dir, refs, MODES, "held-out test")
    val_summary = summarize(val_raw, direction)
    test_summary = summarize(test_raw, direction)
    val_raw.to_csv(args.output / "validation_raw.csv", index=False)
    test_raw.to_csv(args.output / "test_raw.csv", index=False)
    val_summary.to_csv(args.output / "validation_summary.csv", index=False)
    test_summary.to_csv(args.output / "test_summary.csv", index=False)
    protocol = {"train_images": len(train), "validation_images": len(validation),
                "test_images": len(test), "template_regions_by_position": counts,
                "direction_selected_only_on_validation": direction,
                "validation_raw_distance_auc": auc_positive,
                "distance": "initial-manuscript background-denominator chi-square",
                "calibrated_weight": "empirical monotone rank of direction * chi-square distance",
                "online_update": False, "test_labels_used_for_calibration": False}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
