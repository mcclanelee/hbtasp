"""Read-only structural and scientific-contract checks for the release."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "experiments" / "checkpoints"

EXPECTED_ROWS = {
    "v4_r2_6_hidden_overrun": ("cell_results.csv", 100),
    "v5_productive_cooling_ablation": ("cell_results.csv", 560),
    "v5_strict_all_regions_boundary": ("cell_results.csv", 700),
    "v5_tight_period_dynamic_boundary": ("cell_results.csv", 700),
    "v8_calibrated_final_factorial": ("cell_results.csv", 800),
    "v8_calibrated_tight_boundary": ("cell_results.csv", 700),
    "v8_calibrated_multidefect_stratified": ("cell_strata.csv", 1200),
    "v9_thermal_augmented_factorial": ("cell_results.csv", 800),
    "v10_mandatory_outcome_audit": ("cell_results.csv", 800),
    "v11_unified_overall": ("cell_results.csv", 800),
    "v12_protected_certificate_audit": ("certificate_instances.csv", 40),
    "v13_esatd_unified_levels": ("cell_results.csv", 1000),
    "v13_full_image_t4_boundary": ("cell_results.csv", 400),
    "v13_heat_paper_aligned": ("cell_results.csv", 1000),
    "v14_strict_all_regions_matched": ("cell_results.csv", 200),
    "v15_heat_100ms_feasible_domain": ("cell_results.csv", 40),
    "v17_thermal_deployment_sensitivity": ("cell_results.csv", 80),
    "v18_wide_cooling_mechanism": ("cell_results.csv", 880),
    "v19_topk_mandatory_sweep": ("cell_results.csv", 800),
    "v21_multicue_priority_aligned": ("cell_results.csv", 600),
    "v22_priority_discrimination_envelope": ("cell_results.csv", 600),
    "scheduler_overhead": ("reported_mean_overhead.csv", 25),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def check_close(
    failures: list[str], label: str, observed: float, expected: float,
    tolerance: float = 5e-12,
) -> None:
    """Check a frozen published value without hiding rounding drift."""
    if abs(observed - expected) > tolerance:
        failures.append(
            f"{label}: observed={observed:.12g}, expected={expected:.12g}"
        )


def main() -> None:
    failures: list[str] = []
    loaded: dict[str, list[dict[str, str]]] = {}

    for name, (filename, expected) in EXPECTED_ROWS.items():
        path = CHECKPOINTS / name / filename
        if not path.is_file():
            failures.append(f"{name}: missing {filename}")
            continue
        records = read_rows(path)
        loaded[name] = records
        if len(records) != expected:
            failures.append(f"{name}: rows={len(records)}, expected={expected}")
        checkpoint = CHECKPOINTS / name / "checkpoint.json"
        if checkpoint.is_file():
            metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
            declared = metadata.get("validated_cells", metadata.get("completed_cells"))
            if declared is not None and int(declared) != expected:
                failures.append(
                    f"{name}: checkpoint cells={declared}, expected={expected}"
                )

    residual_grids = {
        "v13_esatd_unified_levels": "mandatory_terminal_residual",
        "v13_full_image_t4_boundary": "mandatory_terminal_residual",
        "v13_heat_paper_aligned": "mandatory_terminal_residual",
        "v14_strict_all_regions_matched": "terminal_residual",
        "v15_heat_100ms_feasible_domain": "mandatory_terminal_residual",
        "v18_wide_cooling_mechanism": "mandatory_terminal_residual",
        "v19_topk_mandatory_sweep": "mandatory_terminal_residual",
        "v21_multicue_priority_aligned": "mandatory_terminal_residual",
        "v22_priority_discrimination_envelope": "mandatory_terminal_residual",
    }
    for name, field in residual_grids.items():
        if any(abs(numeric(row, field)) > 1e-12 for row in loaded.get(name, [])):
            failures.append(f"{name}: nonzero terminal-state residual")

    nominal = loaded.get("v8_calibrated_final_factorial", [])
    if any(numeric(row, "thermal_violations") != 0 for row in nominal):
        failures.append("v8 nominal grid: thermal violations must be zero")

    thermal = loaded.get("v9_thermal_augmented_factorial", [])
    if any(numeric(row, "iit_celsius_seconds") > 1e-12 for row in thermal):
        failures.append("v9 nominal reconstruction: IIT must be zero")
    if any(numeric(row, "peak_temperature_c") > 60.0 + 1e-8 for row in thermal):
        failures.append("v9 nominal reconstruction: peak temperature exceeds 60 C")

    topk = loaded.get("v19_topk_mandatory_sweep", [])
    if topk and sorted({int(numeric(row, "mandatory_count")) for row in topk}) != [1, 2, 3, 4]:
        failures.append("v19: mandatory-count levels must be 1, 2, 3, and 4")

    priority = loaded.get("v22_priority_discrimination_envelope", [])
    expected_strategies = {"random_frozen", "calibrated_histogram", "defect_oracle"}
    if priority and {row.get("strategy") for row in priority} != expected_strategies:
        failures.append("v22: unexpected priority-strategy set")
    if any(abs(numeric(row, "priority_overhead_ms")) > 1e-12 for row in priority):
        failures.append("v22: controlled discrimination envelope must use zero latency")
    if any(int(numeric(row, "mandatory_count")) != 1 for row in priority):
        failures.append("v22: controlled discrimination envelope must use Top-1")

    terminal_audit = loaded.get("v10_mandatory_outcome_audit", [])
    terminal_fields = [
        "mandatory_completed_on_time",
        "mandatory_completed_late",
        "mandatory_expired",
        "mandatory_dispatch_infeasible",
        "mandatory_allocation_rejected",
    ]
    for row in terminal_audit:
        accounted = sum(numeric(row, field) for field in terminal_fields)
        if abs(accounted - numeric(row, "mandatory_released")) > 1e-12:
            failures.append("v10: mandatory terminal outcomes are not exhaustive")
            break

    # The article reports equal-cell means, not release-count-weighted rates.
    outcome_summary_path = (
        CHECKPOINTS / "v10_mandatory_outcome_audit" /
        "mandatory_outcome_summary.csv"
    )
    if outcome_summary_path.is_file():
        outcome_summary = {
            row["configuration"]: row for row in read_rows(outcome_summary_path)
        }
        published_cell_means = {
            "EDF-FixedL3": 0.25945166666666664,
            "EDF-Dynamic-Reservation": 0.212706125,
            "HBTASP-FixedL3": 0.0,
            "HBTASP-Dynamic": 0.0325,
        }
        for configuration, expected in published_cell_means.items():
            row = outcome_summary.get(configuration)
            if row is None:
                failures.append(f"v10: missing configuration {configuration}")
                continue
            check_close(
                failures,
                f"v10 {configuration} equal-cell service failure",
                numeric(row, "cell_mean_mandatory_service_failure_rate"),
                expected,
            )

    # Supplement S8: 560-cell paired low-voltage ablation.
    cooling_path = (
        CHECKPOINTS / "v5_productive_cooling_ablation" /
        "paired_seed_contrasts.csv"
    )
    if cooling_path.is_file():
        cooling = {
            row["metric"]: row for row in read_rows(cooling_path)
            if row.get("stratum") == "all_periods"
        }
        published_cooling = {
            "mean_voltage": -0.004677221149535851,
            "peak_modeled_temperature": -0.00014782587708024,
            "pixel_defect_recall": 0.001141328963295343,
            "image_complete_miss_rate": -0.0008075892857142832,
            "mean_complete_image_dice": -0.002363985056488706,
        }
        for metric, expected in published_cooling.items():
            row = cooling.get(metric)
            if row is None:
                failures.append(f"v5 cooling: missing metric {metric}")
                continue
            check_close(
                failures,
                f"v5 cooling {metric}",
                numeric(row, "cooling_minus_control"),
                expected,
            )

    # Supplement S7: independent 500-run T4 overlap-processing audit.
    overlap_runtime_path = (
        ROOT / "perception_evidence" / "overlap_runtime_t4" /
        "overlap_runtime_t4.json"
    )
    if overlap_runtime_path.is_file():
        overlap = json.loads(overlap_runtime_path.read_text(encoding="utf-8"))
        geometries = overlap.get("geometries", {})
        published_overlap = {
            "non_overlap_4x400": (54.76820860000001, 63.34948849999998),
            "mild_overlap_4x416": (55.1576648, 64.13645849999999),
            "heavy_overlap_5x400": (67.36229939999998, 81.39100099999997),
        }
        for geometry, (mean_ms, p995_ms) in published_overlap.items():
            row = geometries.get(geometry)
            if row is None:
                failures.append(f"overlap runtime: missing geometry {geometry}")
                continue
            if int(row.get("runs", 0)) != 500:
                failures.append(f"overlap runtime {geometry}: runs must equal 500")
            check_close(
                failures, f"overlap runtime {geometry} mean",
                float(row["mean_ms"]), mean_ms,
            )
            check_close(
                failures, f"overlap runtime {geometry} p99.5",
                float(row["p99_5_ms"]), p995_ms,
            )

    strict = loaded.get("v5_strict_all_regions_boundary", [])
    if any(row.get("hardware_input_gpu1") != "power_capped_T4_measured_input"
           for row in strict):
        failures.append("v5 strict grid: GPU1 provenance must identify the measured power-capped T4 profile")

    full_profile_path = ROOT / "FULL_IMAGE_T4_PROFILE_LOCK_V1.json"
    if full_profile_path.is_file():
        full_profile = json.loads(full_profile_path.read_text(encoding="utf-8"))
        for model_name, model in full_profile.get("models", {}).items():
            if "gpu1_power_capped_profile_ms" not in model:
                failures.append(f"full-image profile: missing measured GPU1 profile for {model_name}")
            if "measured T4 profile" not in model.get("gpu1_provenance", ""):
                failures.append(f"full-image profile: incorrect GPU1 provenance for {model_name}")

    overall_protocol_path = ROOT / "experiments" / "OVERALL_PROTOCOL_V11.json"
    if overall_protocol_path.is_file():
        platform = json.loads(overall_protocol_path.read_text(encoding="utf-8"))["platform"]
        if "gpu1_power_capped_measured_wcet_ms_l1_l5" not in platform:
            failures.append("Overall protocol: measured power-capped GPU1 vector is missing")

    required = [
        full_profile_path,
        ROOT / "experiments" / "HEAT_PAPER_CONTRACT_V1.md",
        ROOT / "experiments" / "DATA_AUTHORITY_V5.md",
        ROOT / "experiments" / "checkpoints" / "v5_multicue_priority" / "multicue_test_pool.json",
        ROOT / "experiments" / "checkpoints" / "v9_restored_main_figures" / "v9_factorial_ablation.pdf",
        ROOT / "experiments" / "checkpoints" / "v10_mandatory_outcome_audit" / "v10_mandatory_terminal_outcomes.pdf",
        ROOT / "experiments" / "checkpoints" / "v10_mandatory_outcome_audit" / "v10_dynamic_hard_realtime_service_map.pdf",
        ROOT / "experiments" / "checkpoints" / "v5_productive_cooling_ablation" / "paired_seed_contrasts.csv",
        ROOT / "experiments" / "checkpoints" / "v5_productive_cooling_ablation" / "r2_1_productive_cooling_ablation.pdf",
        ROOT / "experiments" / "checkpoints" / "v14_unified_overall_corrected_heat" / "cell_results.csv",
        ROOT / "experiments" / "checkpoints" / "v19_topk_mandatory_sweep" / "v19_topk_load_envelope.pdf",
        ROOT / "experiments" / "checkpoints" / "v21_multicue_priority_aligned" / "v21_priority_calibration_quality_cost.pdf",
        ROOT / "experiments" / "checkpoints" / "scheduler_overhead" / "reported_mean_overhead.csv",
        ROOT / "perception_evidence" / "overlap_corrected_final" / "summary.csv",
        overlap_runtime_path,
        ROOT / "perception_evidence" / "overlap_runtime_t4" / "PROTOCOL_AND_RESULTS.md",
    ]
    failures.extend(
        f"missing required artifact: {path.relative_to(ROOT).as_posix()}"
        for path in required if not path.is_file()
    )

    if failures:
        raise SystemExit("RELEASE VERIFICATION FAILED\n" + "\n".join(failures))
    print(
        f"RELEASE VERIFICATION PASSED: {len(EXPECTED_ROWS)} experiment/result tables "
        "and the declared scientific contracts"
    )


if __name__ == "__main__":
    main()
