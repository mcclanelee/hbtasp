"""Cluster-bootstrap analysis for the ground-truth-defined boundary subset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from publication_style import apply_publication_style, BLUE, GREEN, ORANGE
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "perception_evidence/boundary_subset_corrected_final"
ARMS = ["non_overlap_4x400", "mild_overlap_4x416", "heavy_overlap_5x400"]
LABELS = ["Non-overlap", "Mild overlap", "Heavy overlap"]
COLORS = [BLUE, GREEN, ORANGE]
BOOTSTRAPS = 10_000


def aggregate(frame: pd.DataFrame) -> dict[str, float]:
    tp, fp, fn = frame[["tp", "fp", "fn"]].sum()
    return {
        "roi_dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "pixel_recall": tp / max(1, tp + fn),
        "any_overlap_recall": frame.any_overlap.sum() / len(frame),
        "iou10_recall": frame.iou10.sum() / len(frame),
    }


def main() -> None:
    components = {arm: pd.read_csv(OUT / f"{arm}_components.csv") for arm in ARMS}
    images = {arm: pd.read_csv(OUT / f"{arm}_images.csv") for arm in ARMS}
    keys = ["image_id", "class_index", "component_label"]
    reference = components[ARMS[0]][keys]
    for arm in ARMS[1:]:
        if not reference.equals(components[arm][keys]):
            raise RuntimeError(f"component subset changed across arms: {arm}")
    image_ids = np.asarray(sorted(reference.image_id.unique()))
    if len(image_ids) != 468 or len(reference) != 641:
        raise RuntimeError("unexpected boundary subset size")

    observed = {arm: aggregate(components[arm]) for arm in ARMS}
    for arm in ARMS:
        observed[arm]["image_complete_miss_rate"] = images[arm].complete_miss.mean()

    rng = np.random.default_rng(69)
    # One row per image cluster: tp, fp, fn, component count, hits, IoU hits.
    cluster_arrays = {}
    for arm, frame in components.items():
        grouped = frame.groupby("image_id").agg(
            tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
            component_count=("component_label", "size"),
            any_hits=("any_overlap", "sum"), iou10_hits=("iou10", "sum"),
        ).reindex(image_ids)
        cluster_arrays[arm] = grouped.to_numpy(dtype=float)
    image_miss = {arm: images[arm].set_index("image_id").complete_miss for arm in ARMS}
    rows = []
    for arm in ARMS[1:]:
        boot = {metric: [] for metric in observed[arm]}
        for _ in range(BOOTSTRAPS):
            sample_index = rng.integers(0, len(image_ids), size=len(image_ids))
            av = cluster_arrays[arm][sample_index].sum(axis=0)
            bv = cluster_arrays[ARMS[0]][sample_index].sum(axis=0)
            def values(v):
                tp, fp, fn, count, hits, iou_hits = v
                return {"roi_dice": 2*tp/max(1,2*tp+fp+fn),
                        "pixel_recall": tp/max(1,tp+fn),
                        "any_overlap_recall": hits/max(1,count),
                        "iou10_recall": iou_hits/max(1,count)}
            aa, bb = values(av), values(bv)
            for metric in aa:
                boot[metric].append(aa[metric] - bb[metric])
            boot["image_complete_miss_rate"].append(
                image_miss[arm].reindex(image_ids).to_numpy()[sample_index].mean()
                - image_miss[ARMS[0]].reindex(image_ids).to_numpy()[sample_index].mean())
        for metric in observed[arm]:
            delta = observed[arm][metric] - observed[ARMS[0]][metric]
            low, high = np.quantile(boot[metric], [.025, .975])
            rows.append({"contrast": f"{arm} - {ARMS[0]}", "metric": metric,
                         "difference": delta, "cluster_bootstrap_ci95_low": low,
                         "cluster_bootstrap_ci95_high": high,
                         "bootstrap_samples": BOOTSTRAPS})
    contrasts = pd.DataFrame(rows)

    # Exact paired test for image-level complete misses.
    tests = []
    base_miss = image_miss[ARMS[0]].sort_index()
    for arm in ARMS[1:]:
        other = image_miss[arm].sort_index()
        improved = int(((base_miss == 1) & (other == 0)).sum())
        worsened = int(((base_miss == 0) & (other == 1)).sum())
        discordant = improved + worsened
        p = 1.0 if discordant == 0 else stats.binomtest(
            min(improved, worsened), discordant, .5, alternative="two-sided").pvalue
        tests.append({"contrast": f"{arm} - {ARMS[0]}",
                      "baseline_miss_resolved": improved,
                      "new_miss_introduced": worsened,
                      "mcnemar_exact_p": p})
    pd.DataFrame(tests).to_csv(OUT / "paired_image_miss_tests.csv", index=False)
    contrasts.to_csv(OUT / "cluster_bootstrap_contrasts.csv", index=False)
    pd.DataFrame([{"partition": arm, **observed[arm]} for arm in ARMS]).to_csv(
        OUT / "validated_boundary_metrics.csv", index=False)

    panels = [("roi_dice", "Boundary ROI Dice"),
              ("pixel_recall", "Boundary pixel recall"),
              ("iou10_recall", r"Component recall ($IoU\geq0.1$)"),
              ("image_complete_miss_rate", "Boundary-image miss rate")]
    apply_publication_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.75), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        values = [observed[arm][metric] for arm in ARMS]
        bars = ax.bar(range(3), values, width=.46, color=COLORS, alpha=.92,
                      edgecolor="#333333", linewidth=.75)
        ax.set_xticks(range(3), LABELS, rotation=25, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
        bottom = max(0, min(values) - .08)
        top = max(values) + max(.035, (max(values) - bottom) * .30)
        ax.set_ylim(bottom, top)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + (top-bottom)*.025,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=10.8,
                    fontweight="semibold")
    fig.savefig(OUT / "r2_4_boundary_subset.pdf", bbox_inches="tight")
    fig.savefig(OUT / "r2_4_boundary_subset.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    c = contrasts.set_index(["contrast", "metric"])
    heavy = f"{ARMS[2]} - {ARMS[0]}"
    report = f"""# Dedicated boundary-defect subset

The subset was fixed from ground truth before predictions were inspected and
contains 468 images and 641 class-specific connected components crossing an
original cut at x=400, 800, or 1200. Confidence intervals use 10,000 paired
cluster bootstrap resamples at image level.

Heavy overlap versus non-overlap changes ROI Dice by
{c.loc[(heavy,'roi_dice'),'difference']:+.5f} (95% CI
{c.loc[(heavy,'roi_dice'),'cluster_bootstrap_ci95_low']:+.5f} to
{c.loc[(heavy,'roi_dice'),'cluster_bootstrap_ci95_high']:+.5f}), pixel recall by
{c.loc[(heavy,'pixel_recall'),'difference']:+.5f}, and IoU>=0.1 component recall
by {c.loc[(heavy,'iou10_recall'),'difference']:+.5f}. Boundary-image complete
miss rate changes by
{c.loc[(heavy,'image_complete_miss_rate'),'difference']:+.5f}. These benefits
must be weighed against the separately measured prototype T4 p99 processing-time increase;
they do not establish universal superiority.
"""
    (OUT / "ANALYSIS.md").write_text(report, encoding="utf-8")
    details_path = OUT / "details.json"
    details = json.loads(details_path.read_text(encoding="utf-8"))
    details["analysis_status"] = "complete_validated"
    details["bootstrap_samples"] = BOOTSTRAPS
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(pd.DataFrame([{"partition": arm, **observed[arm]} for arm in ARMS]))
    print(contrasts)
    print(pd.DataFrame(tests))


if __name__ == "__main__":
    main()
