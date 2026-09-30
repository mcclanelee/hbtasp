"""Train and evaluate a template-free lightweight regional priority scorer.

The scorer is fitted only on the source-disjoint training split, its single
regularization hyperparameter is selected on the calibration split, and the
test split is touched once.  It is a deployable diagnostic replacement for the
initial gray-histogram priority, not a modification of HBTASP scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from perception_code.train_csdnn_shared_corrected import prepare_frame
from perception_code.train_evaluate_deeplab_corrected import full_truth

REGIONS = ((0, 400), (400, 800), (800, 1200), (1200, 1600))
CS = (0.01, 0.1, 1.0, 10.0)
FEATURE_VERSION = "multicue_v1_gray_lab_gradient_spatial_relative"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled4(array: np.ndarray) -> np.ndarray:
    return cv2.resize(array, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)


def image_features(image: np.ndarray) -> np.ndarray:
    """Four regional feature vectors; no background template or test label."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    vectors, scalar_rows = [], []
    for start, end in REGIONS:
        g = cv2.resize(gray[:, start:end], (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        color = cv2.resize(lab[:, start:end], (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        mean, std = float(g.mean()), float(g.std() + 1e-6)
        normalized = (g - mean) / std
        gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        laplacian = cv2.Laplacian(normalized, cv2.CV_32F)
        angle = (cv2.phase(gx, gy, angleInDegrees=False) % np.pi)
        orientation, _ = np.histogram(angle, bins=8, range=(0, np.pi), weights=magnitude)
        orientation = orientation.astype(np.float32) / max(1e-6, orientation.sum())
        q10, q90 = np.quantile(g, [.10, .90])
        scalars = np.asarray([
            mean / 255.0, std / 255.0, q10 / 255.0, q90 / 255.0,
            float(magnitude.mean()), float(magnitude.std()),
            float(np.abs(laplacian).mean()), float((magnitude > 1.0).mean()),
            float(color[..., 1].mean() / 255.0), float(color[..., 1].std() / 255.0),
            float(color[..., 2].mean() / 255.0), float(color[..., 2].std() / 255.0),
        ], np.float32)
        scalar_rows.append(scalars)
        vectors.append(np.concatenate([scalars, orientation, pooled4(normalized),
                                       pooled4(magnitude)]).astype(np.float32))
    scalar_array = np.stack(scalar_rows)
    relative = scalar_array - np.median(scalar_array, axis=0, keepdims=True)
    return np.stack([np.concatenate([vectors[i], relative[i]]) for i in range(4)])


def extract_split(ids, frame, image_dir: Path, cache: Path):
    if cache.exists():
        loaded = np.load(cache, allow_pickle=False)
        return loaded["x"], loaded["y"], loaded["pixels"], loaded["ids"]
    xs, ys, pixels, used_ids = [], [], [], []
    for index, image_id in enumerate(ids):
        image = cv2.imread(str(image_dir / image_id))
        if image is None:
            raise FileNotFoundError(image_dir / image_id)
        truth = full_truth(frame.loc[image_id])
        region_pixels = np.asarray([truth[:, :, a:b].sum() for a, b in REGIONS], np.int64)
        xs.append(image_features(image)); ys.append(region_pixels > 0)
        pixels.append(region_pixels); used_ids.append(image_id)
        if (index + 1) % 250 == 0:
            print(f"features {cache.stem}: {index+1}/{len(ids)}", flush=True)
    x = np.stack(xs); y = np.stack(ys); pixel_array = np.stack(pixels)
    id_array = np.asarray(used_ids)
    np.savez_compressed(cache, x=x, y=y, pixels=pixel_array, ids=id_array)
    return x, y, pixel_array, id_array


def ranking_metrics(probability, labels, pixels):
    chosen = probability.argmax(axis=1)
    rows = np.arange(len(chosen))
    total_pixels = pixels.sum(axis=1)
    return {
        "region_roc_auc": float(roc_auc_score(labels.reshape(-1), probability.reshape(-1))),
        "region_average_precision": float(average_precision_score(
            labels.reshape(-1), probability.reshape(-1))),
        "top1_defect_region_hit_rate": float(labels[rows, chosen].mean()),
        "priority_defect_pixel_share": float(np.mean(
            pixels[rows, chosen] / np.maximum(1, total_pixels))),
        "priority_complete_miss_rate": float((pixels[rows, chosen] == 0).mean()),
    }


def score_image(model, image):
    probability = model.predict_proba(image_features(image))[:, 1]
    total = probability.sum()
    return probability / total if total > 0 else np.full(4, .25)


def benchmark(model, ids, image_dir: Path):
    images = [cv2.imread(str(image_dir / image_id)) for image_id in ids[:40]]
    rng = np.random.default_rng(69)
    result = {}
    for batch_size in (1, 4, 8, 10):
        for _ in range(5):
            for image in images[:batch_size]: score_image(model, image)
        values = []
        for _ in range(100):
            selected = rng.integers(0, len(images), size=batch_size)
            start = time.perf_counter()
            for i in selected: score_image(model, images[i])
            values.append((time.perf_counter() - start) * 1000)
        array = np.asarray(values)
        result[str(batch_size)] = {"runs": len(values), "mean_ms": float(array.mean()),
                                   "p99_ms": float(np.percentile(array, 99)),
                                   "max_ms": float(array.max())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--histogram-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = args.project / "data" / "severstal-steel-defect-detection"
    image_dir = data / "train_images"
    frame = prepare_frame(data / "train.csv")
    split = json.loads(args.split_protocol.read_text(encoding="utf-8"))
    arrays = {}
    for name in ("train", "calibration", "test"):
        arrays[name] = extract_split(split[f"{name}_ids"], frame, image_dir,
                                     args.output / f"{name}_features.npz")
    x_train, y_train, _, _ = arrays["train"]
    x_cal, y_cal, _, _ = arrays["calibration"]
    selected_c, selected_ap = None, -np.inf
    selection = []
    for c in CS:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, random_state=69))
        model.fit(x_train.reshape(-1, x_train.shape[-1]), y_train.reshape(-1))
        probability = model.predict_proba(x_cal.reshape(-1, x_cal.shape[-1]))[:, 1]
        ap = average_precision_score(y_cal.reshape(-1), probability)
        selection.append({"C": c, "calibration_average_precision": float(ap)})
        if ap > selected_ap: selected_c, selected_ap = c, ap
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=selected_c, class_weight="balanced", max_iter=2000, random_state=69))
    model.fit(x_train.reshape(-1, x_train.shape[-1]), y_train.reshape(-1))
    joblib.dump(model, args.output / "multicue_priority_model.joblib")

    x_test, y_test, pixels_test, test_ids = arrays["test"]
    probability = model.predict_proba(x_test.reshape(-1, x_test.shape[-1]))[:, 1].reshape(-1, 4)
    metrics = ranking_metrics(probability, y_test, pixels_test)
    id_to_index = {image_id: i for i, image_id in enumerate(test_ids.tolist())}
    histogram_pool = json.loads(args.histogram_pool.read_text(encoding="utf-8"))
    common_indices = np.asarray([id_to_index[item["image_id"]] for item in histogram_pool])
    common_multicue = ranking_metrics(probability[common_indices], y_test[common_indices],
                                      pixels_test[common_indices])
    hist_choice = np.asarray([int(item["mandatory_idx"]) for item in histogram_pool])
    common_pixels = pixels_test[common_indices]
    row = np.arange(len(hist_choice))
    histogram_metrics = {
        "top1_defect_region_hit_rate": float((common_pixels[row, hist_choice] > 0).mean()),
        "priority_defect_pixel_share": float(np.mean(
            common_pixels[row, hist_choice] / np.maximum(1, common_pixels.sum(axis=1)))),
        "priority_complete_miss_rate": float((common_pixels[row, hist_choice] == 0).mean()),
    }
    replay_pool = []
    for source, index in zip(histogram_pool, common_indices):
        weights = probability[index]
        weights = weights / weights.sum() if weights.sum() else np.full(4, .25)
        replay_pool.append({**source, "weights": weights.tolist(),
                            "mandatory_idx": int(weights.argmax()),
                            "priority_protocol": FEATURE_VERSION})
    (args.output / "multicue_test_pool.json").write_text(
        json.dumps(replay_pool, indent=2), encoding="utf-8")
    timing = benchmark(model, test_ids.tolist(), image_dir)
    report = {
        "status": "complete", "feature_version": FEATURE_VERSION,
        "split_protocol": str(args.split_protocol), "selected_C": selected_c,
        "hyperparameter_selection": selection, "test_metrics_1334_images": metrics,
        "common_500_multicue": common_multicue,
        "common_500_histogram": histogram_metrics,
        "cpu_priority_latency_ms_excluding_image_io": timing,
        "measurement_hardware": "T4_experiment_host_CPU_measurement",
        "host_gpu_context": "NVIDIA_T4_GPU_not_used_in_timed_region",
        "timing_scope": "CPU_feature_extraction_and_classifier_only_images_preloaded",
        "gpu_timing_status": "GPU_not_used_in_timed_region",
        "no_online_clean_template": True,
        "sha256": {"split": sha256(args.split_protocol),
                   "histogram_pool": sha256(args.histogram_pool),
                   "model": sha256(args.output / "multicue_priority_model.joblib"),
                   "replay_pool": sha256(args.output / "multicue_test_pool.json")},
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
