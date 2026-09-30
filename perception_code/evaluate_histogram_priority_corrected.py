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

try:
    from perception_code.train_csdnn_shared_corrected import prepare_frame, rle_to_mask
except ModuleNotFoundError:  # Direct execution from perception_code/.
    from train_csdnn_shared_corrected import prepare_frame, rle_to_mask


def histogram(gray, bins=64):
    value = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    return value / max(1.0, value.sum())


def chi_square(value, reference, eps=1e-9):
    # Symmetric chi-square is numerically stable and does not privilege either
    # sample when a bin is empty.
    return 0.5 * np.sum((value - reference) ** 2 / (value + reference + eps))


def perturb(gray, mode):
    if mode == "original":
        return gray
    if mode == "bright+20":
        return cv2.convertScaleAbs(gray, alpha=1.0, beta=20)
    if mode == "bright-20":
        return cv2.convertScaleAbs(gray, alpha=1.0, beta=-20)
    if mode == "contrast+20%":
        return cv2.convertScaleAbs(gray, alpha=1.2, beta=-25.5)
    if mode == "contrast-20%":
        return cv2.convertScaleAbs(gray, alpha=.8, beta=25.5)
    raise ValueError(mode)


def metrics(frame):
    y = frame.defective.to_numpy(int); score = frame.score.to_numpy(float)
    rho, rho_p = spearmanr(score, frame.defect_pixels.to_numpy(float))
    threshold = np.quantile(score, .25)
    low = frame[frame.score <= threshold]
    return {"regions": len(frame), "defective_regions": int(y.sum()),
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "prevalence": float(y.mean()), "spearman_area": float(rho),
            "spearman_p": float(rho_p), "lowest_quartile_regions": len(low),
            "lowest_quartile_defect_rate": float(low.defective.mean()),
            "lowest_quartile_defective_regions": int(low.defective.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--split-protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-template-regions", type=int, default=2000)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = args.project / "data" / "severstal-steel-defect-detection"
    image_dir = data / "train_images"; frame = prepare_frame(data / "train.csv")
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    train, test = frame.loc[split["train_ids"]], frame.loc[split["test_ids"]]
    templates = [[] for _ in range(4)]
    for image_id, row in tqdm(train.iterrows(), total=len(train), desc="clean-region templates"):
        image = cv2.imread(str(image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        masks = [rle_to_mask(row[f"class_{c}"]) for c in range(1, 5)]
        combined = np.logical_or.reduce(masks)
        for region in range(4):
            if len(templates[region]) >= args.max_template_regions:
                continue
            x0, x1 = region * 400, (region + 1) * 400
            if not combined[:, x0:x1].any():
                templates[region].append(histogram(image[:, x0:x1]))
        if all(len(x) >= args.max_template_regions for x in templates):
            break
    references = [np.mean(x, axis=0) for x in templates]
    modes = ["original", "bright+20", "bright-20", "contrast+20%", "contrast-20%"]
    rows = []
    for image_id, row in tqdm(test.iterrows(), total=len(test), desc="test histogram priority"):
        image = cv2.imread(str(image_dir / image_id), cv2.IMREAD_GRAYSCALE)
        combined = np.logical_or.reduce([rle_to_mask(row[f"class_{c}"])
                                         for c in range(1, 5)])
        for region in range(4):
            x0, x1 = region * 400, (region + 1) * 400
            pixels = int(combined[:, x0:x1].sum())
            crop = image[:, x0:x1]
            for mode in modes:
                rows.append({"image_id": image_id, "region": region,
                             "mode": mode, "defect_pixels": pixels,
                             "defective": int(pixels > 0),
                             "score": chi_square(histogram(perturb(crop, mode)),
                                                 references[region])})
    raw = pd.DataFrame(rows); raw.to_csv(args.output / "histogram_priority_raw.csv", index=False)
    summary = []
    for mode, cell in raw.groupby("mode"):
        record = {"mode": mode, **metrics(cell)}
        # Does the highest score in an image select any defective region?
        eligible = cell.groupby("image_id").defective.max()
        eligible = eligible[eligible > 0].index
        top = cell[cell.image_id.isin(eligible)].sort_values(
            ["image_id", "score"], ascending=[True, False]).groupby("image_id").first()
        record["top1_defective_region_hit_rate"] = float(top.defective.mean())
        # Oracle denominator: fraction of eligible images with >1 defective region.
        counts = cell[cell.image_id.isin(eligible)].groupby("image_id").defective.sum()
        record["multi_defect_region_image_rate"] = float((counts > 1).mean())
        summary.append(record)
    pd.DataFrame(summary).to_csv(args.output / "histogram_priority_summary.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({
        "train_ids": len(train), "test_ids": len(test),
        "template_regions_by_position": [len(x) for x in templates],
        "bins": 64, "template_source": "label-confirmed clean training regions",
        "online_update": False, "score": "symmetric chi-square"}, indent=2),
        encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
