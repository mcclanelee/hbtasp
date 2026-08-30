"""Protected-mode audit for the sufficient Stage-I timing certificate.

This is a theorem-correspondence harness, not the primary graceful-replay
factorial.  Each production line is one implicit-deadline periodic mandatory
stream.  A stream keeps a fixed processor mapping within an audit cell.
Admission directly tests phi_j = U_j^M + B_j / D_min <= 1.  The replay then
injects the conservative one-segment carry-in B_j at every mandatory busy
interval and uses strict mandatory priority with EDF within the class.

This harness is deliberately estimator-agnostic: it audits abstract mandatory
streams after designation and does not reconstruct image-region identities.
Under the evaluated contract every mandatory region uses the same utility-best
L5 execution profile, so replacing raw with calibrated priority changes which
region is designated but not the certificate inputs of this correspondence
harness. It must not be described as a perception or priority-quality audit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.initial_manuscript_hbtasp import GPU_WCET


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/checkpoints/v12_protected_certificate_audit"
PERIODS_MS = (100, 150, 200, 250, 300)
LINE_COUNTS = (4, 6, 8, 10)
EPOCHS = 1000
MANDATORY_LEVEL = 4  # utility-best L5 in the historical scheduling profile
TOL = 1e-12


def certificate(processor: int, streams: int, period_s: float) -> dict:
    c = GPU_WCET[processor][MANDATORY_LEVEL]
    # Safe theorem value fixed before mandatory admission: the maximum baseline
    # execution bound over the complete mandatory/optional candidate-profile
    # universe eligible on this processor, including profiles not ultimately
    # admitted. GPU_WCET is monotone across L1--L5 on both processors, so this
    # complete-universe maximum is attained by L5. Stage II enforces
    # C_selected <= mu, hence no later path selection enlarges this bound.
    b = GPU_WCET[processor][MANDATORY_LEVEL]
    utilization = streams * c / period_s
    phi = utilization + b / period_s
    return {
        "processor": processor,
        "accepted_streams": streams,
        "wcet_s": c,
        "blocking_bound_s": b,
        "d_min_s": period_s,
        "mandatory_utilization": utilization,
        "phi": phi,
        "margin": 1.0 - phi,
        "certificate_pass": phi <= 1.0 + TOL,
    }


def assign_streams(lines: int, period_s: float) -> tuple[list[int], int]:
    assigned = [0, 0]
    rejected = 0
    for _stream in range(lines):
        candidates = []
        for processor in range(2):
            trial = certificate(processor, assigned[processor] + 1, period_s)
            if trial["certificate_pass"]:
                candidates.append((trial["phi"], processor))
        if not candidates:
            rejected += 1
        else:
            _phi, processor = min(candidates)
            assigned[processor] += 1
    return assigned, rejected


def replay_instance(instance: dict) -> dict:
    # At every synchronous release, one lower-priority non-preemptive segment
    # of length B_j is conservatively assumed to carry into the mandatory busy
    # interval. Strict mandatory priority prevents any further optional start.
    demand = (instance["blocking_bound_s"]
              + instance["accepted_streams"] * instance["wcet_s"])
    late_per_epoch = int(demand > instance["d_min_s"] + TOL)
    released = instance["accepted_streams"] * EPOCHS
    late = late_per_epoch * instance["accepted_streams"] * EPOCHS
    return {
        "replay_epochs": EPOCHS,
        "mandatory_released": released,
        "mandatory_late": late,
        "admitted_mandatory_dmr": late / released if released else 0.0,
        "worst_busy_interval_s": demand,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for period_ms in PERIODS_MS:
        period_s = period_ms / 1000.0
        for lines in LINE_COUNTS:
            assigned, rejected = assign_streams(lines, period_s)
            for processor, streams in enumerate(assigned):
                row = {
                    "period_ms": period_ms,
                    "production_lines": lines,
                    "processor": processor,
                    "released_streams": lines,
                    "pre_rejected_streams_cell": rejected,
                    **certificate(processor, streams, period_s),
                }
                row.update(replay_instance(row))
                rows.append(row)

    fields = list(rows[0])
    with (OUT / "certificate_instances.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    active = [r for r in rows if r["accepted_streams"] > 0]
    margins = sorted(r["margin"] for r in active)
    q05 = margins[max(0, int(0.05 * (len(margins) - 1)))]
    summary = {
        "protocol": "protected mandatory priority; EDF within class",
        "certificate_instances": len(active),
        "certificate_pass_rate": sum(r["certificate_pass"] for r in active) / len(active),
        "max_phi": max(r["phi"] for r in active),
        "min_margin": min(margins),
        "median_margin": (margins[(len(margins)-1)//2] + margins[len(margins)//2]) / 2,
        "p05_margin": q05,
        "total_pre_rejected_stream_instances": sum(
            r["pre_rejected_streams_cell"] for r in rows if r["processor"] == 0
        ),
        "mandatory_released": sum(r["mandatory_released"] for r in active),
        "mandatory_late": sum(r["mandatory_late"] for r in active),
        "admitted_mandatory_dmr": (
            sum(r["mandatory_late"] for r in active)
            / sum(r["mandatory_released"] for r in active)
        ),
        "scope": "theorem correspondence only; not a replacement for the graceful factorial",
        "priority_estimator_dependency": "none: abstract post-designation mandatory streams",
        "mandatory_identity_scope": (
            "image-region identities are not represented; all designated mandatory "
            "regions use the same utility-best L5 execution contract"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Protected-mode certificate audit",
        "",
        "This harness audits the sufficient certificate under its own dispatch assumptions; it does not relabel the graceful factorial as theorem validation.",
        "",
        f"- Active processor certificate instances: {summary['certificate_instances']}",
        f"- Certificate pass rate: {summary['certificate_pass_rate']:.4f}",
        f"- Maximum phi: {summary['max_phi']:.6f}",
        f"- Minimum margin: {summary['min_margin']:.6f}",
        f"- Median margin: {summary['median_margin']:.6f}",
        f"- Fifth-percentile margin: {summary['p05_margin']:.6f}",
        f"- Admitted mandatory DMR: {summary['admitted_mandatory_dmr']:.6f}",
        "",
        "A rejected certificate is not evidence of infeasibility because the condition is sufficient, not necessary.",
        "",
        "The harness is estimator-agnostic and operates after mandatory designation. "
        "Raw versus calibrated region identity is therefore outside this audit; under "
        "the evaluated contract every designated mandatory region uses the same L5 "
        "execution profile that supplies the certificate inputs.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
