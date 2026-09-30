"""Measure multi-cue Top-1 priority latency for the aligned 4/6/8/10-line grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np

REGIONS = ((0, 400), (400, 800), (800, 1200), (1200, 1600))


def pooled4(array: np.ndarray) -> np.ndarray:
    return cv2.resize(array, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)


def image_features(image: np.ndarray) -> np.ndarray:
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
        angle = cv2.phase(gx, gy, angleInDegrees=False) % np.pi
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


def score_image(model, image):
    probability = model.predict_proba(image_features(image))[:, 1]
    total = probability.sum()
    return probability / total if total > 0 else np.full(4, .25)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "experiments/checkpoints/v5_multicue_priority/multicue_priority_model.joblib"
FEATURES = ROOT / "experiments/checkpoints/v5_multicue_priority/test_features.npz"
OUT = ROOT / "experiments/checkpoints/v21_multicue_priority_aligned"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    try:
        gpu_names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("A verified NVIDIA T4 experiment host is required") from exc
    if not any(name.strip() == "NVIDIA T4" for name in gpu_names.splitlines()):
        raise RuntimeError("This benchmark must be run on the NVIDIA T4 experiment host")
    OUT.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL)
    ids = np.load(FEATURES, allow_pickle=False)["ids"].tolist()
    images = [cv2.imread(str(args.image_dir / image_id)) for image_id in ids[:40]]
    if any(image is None for image in images):
        raise FileNotFoundError("At least one benchmark image could not be loaded")

    rng = np.random.default_rng(69)
    results = {}
    for batch_size in (4, 6, 8, 10):
        for _ in range(5):
            for image in images[:batch_size]:
                score_image(model, image)
        values = []
        for _ in range(args.runs):
            selected = rng.integers(0, len(images), size=batch_size)
            start = time.perf_counter()
            for index in selected:
                score_image(model, images[index])
            values.append((time.perf_counter() - start) * 1000.0)
        array = np.asarray(values)
        results[str(batch_size)] = {
            "runs": len(values),
            "mean_ms": float(array.mean()),
            "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(array.max()),
            "raw_ms": values,
        }
        print(batch_size, results[str(batch_size)]["p99_ms"], flush=True)

    report = {
        "status": "complete",
        "measurement_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_hardware": "T4_experiment_host_CPU_measurement",
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
        "timing_scope": "CPU feature extraction and classifier scoring; images preloaded",
        "batch_semantics": "one priority decision per simultaneously released image/line",
        "results": results,
        "sha256": {"model": sha256(MODEL), "test_features": sha256(FEATURES)},
    }
    (OUT / "priority_latency_4_6_8_10.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
