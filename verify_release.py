from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    "experiments/checkpoints/v11_unified_overall/cell_results.csv",
    "experiments/checkpoints/v12_protected_certificate_audit/certificate_instances.csv",
    "experiments/checkpoints/v13_full_image_t4_boundary/cell_results.csv",
    "experiments/checkpoints/v17_thermal_deployment_sensitivity/cell_results.csv",
    "experiments/checkpoints/v18_wide_cooling_mechanism/cell_results.csv",
    "experiments/checkpoints/v19_topk_mandatory_sweep/cell_results.csv",
    "experiments/checkpoints/v21_multicue_priority_aligned/cell_results.csv",
    "perception_evidence/overlap_corrected_final/summary.csv",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    manifest = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name != "RELEASE_SHA256.json"
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    }
    (ROOT / "RELEASE_SHA256.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"PASS: {len(manifest)} files verified")


if __name__ == "__main__":
    main()
