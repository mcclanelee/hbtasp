# Published-result consistency audit

Audit date: 2026-09-30

## Scope and outcome

The numerical claims in the published article and Supplementary Material were
checked against the frozen CSV/JSON evidence in this release. The audit covers
the main factorial comparison, unified baselines, mandatory-terminal outcomes,
thermal reconstruction and sensitivity, protected-mode certificate, strict
all-region and full-image boundaries, hidden overruns, low-voltage ablations,
multi-defect strata, Top-K mandatory-region sweep, histogram/multi-cue priority
experiments, controlled priority-discrimination envelope, overlap perception
and timing, and scheduler overhead.

No contradictory published result was found. Reported differences are exact
after applying the article's stated aggregation rule and displayed rounding.
In particular, the service-failure values in the factorial table are
equal-cell means; they must not be replaced by release-count-weighted totals
from the separate terminal-accounting columns.

## Claim-to-evidence checks

| Evidence family | Frozen source | Result |
|---|---|---|
| Four-way factorial comparison | `v8_calibrated_final_factorial/configuration_summary.csv` | Pass |
| Mandatory terminal accounting and feasible-cell maps | `v10_mandatory_outcome_audit/` | Pass; terminal outcomes are exhaustive and article values use equal-cell means |
| Unified Static EDF, ESATD, HEAT, and HBTASP results | `v11_unified_overall/`, `v13_esatd_unified_levels/`, `v13_heat_paper_aligned/` | Pass |
| Nominal and mismatched thermal behavior | `v9_thermal_augmented_factorial/`, `v17_thermal_deployment_sensitivity/` | Pass |
| Protected certificate | `v12_protected_certificate_audit/` | Pass |
| Strict all-region and full-image boundaries | `v14_strict_all_regions_matched/`, `v13_full_image_t4_boundary/`, `v15_heat_100ms_feasible_domain/` | Pass |
| Hidden execution-time overruns | `v4_r2_6_hidden_overrun/` | Pass |
| 560-cell low-voltage paired ablation | `v5_productive_cooling_ablation/` | Pass; voltage, modeled temperature, Dice, recall, and miss effects reproduce the article |
| 880-cell wide-period cooling mechanism | `v18_wide_cooling_mechanism/` | Pass |
| Multi-defect strata | frozen multi-defect summary | Pass |
| Top-K mandatory-region sweep | `v19_topk_mandatory_sweep/` | Pass |
| Multi-cue priority quality/cost | `v21_multicue_priority_aligned/` | Pass |
| Controlled priority-discrimination envelope | `v22_priority_discrimination_envelope/` | Pass |
| Histogram held-out calibration | `perception_evidence/histogram_direction_calibration/` | Pass for the frozen calibrated test summary and perturbation range |
| Full-image DeepLab and YOLO references | `perception_evidence/` and `v13_full_image_t4_boundary/` | Pass |
| Overlap perception and boundary components | `perception_evidence/overlap_corrected_final/` | Pass |
| Independent overlap-processing cost | `perception_evidence/overlap_runtime_t4/` | Pass; 500 runs per crop geometry on NVIDIA T4 |
| Mean scheduling overhead | `scheduler_overhead/reported_mean_overhead.csv` | Pass against the published aggregate table |

## Machine-checked published contracts

`python verify_release.py` additionally locks the following easily confused
claims to their frozen evidence:

- the four equal-cell mandatory-service failure values used in the article;
- all five primary effects in the 560-cell low-voltage ablation; and
- the mean and p99.5 time for each of the three 500-run overlap geometries.

This numerical check complements, rather than replaces, the SHA-256 package
check performed by `python verify_manifest.py`.

## Traceability boundaries

Two results remain aggregate-level evidence rather than fully reconstructable
raw logs. This is a provenance limitation, not a numerical conflict:

1. The original repetition-level scheduler-overhead timing log was not
   retained. The released means reproduce the published table, but dispersion
   and tail latency cannot be recomputed from this package.
2. The region-level raw scores for the uncalibrated histogram direction were
   not retained in the public package. The calibrated held-out summary and
   perturbation results are frozen, while the article's uncalibrated PR-AUC and
   Top-1 values remain reported aggregate results.

Neither aggregate-only item is used as a hard real-time guarantee. The article
already treats mean overhead as descriptive and the histogram score as a
limited-discrimination scheduling rank rather than a defect probability.
