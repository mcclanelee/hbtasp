"""Read-only structural and scientific-contract checks for the release."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "experiments" / "checkpoints"

EXPECTED_ROWS = {
    "v4_r2_6_hidden_overrun": ("cell_results.csv", 100),
    "v5_strict_all_regions_boundary": ("cell_results.csv", 700),
    "v5_tight_period_dynamic_boundary": ("cell_results.csv", 700),
    "v8_calibrated_final_factorial": ("cell_results.csv", 800),
    "v8_calibrated_tight_boundary": ("cell_results.csv", 700),
    "v8_calibrated_multidefect_stratified": ("cell_strata.csv", 1200),
    "v9_thermal_augmented_factorial": ("cell_results.csv", 800),
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
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


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
        ROOT / "perception_evidence" / "overlap_corrected_final" / "summary.csv",
    ]
    failures.extend(
        f"missing required artifact: {path.relative_to(ROOT).as_posix()}"
        for path in required if not path.is_file()
    )

    if failures:
        raise SystemExit("RELEASE VERIFICATION FAILED\n" + "\n".join(failures))
    print(
        f"RELEASE VERIFICATION PASSED: {len(EXPECTED_ROWS)} experiment grids "
        "and the declared scientific contracts"
    )


if __name__ == "__main__":
    main()
