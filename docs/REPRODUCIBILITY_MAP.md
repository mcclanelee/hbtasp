# Reproducibility map

All commands are run from the repository root. Frozen outputs are under
`experiments/checkpoints/`; analyzers validate and summarize those outputs.

| Published article or Supplement evidence | Runner / analyzer | Frozen result |
|---|---|---|
| Main factorial comparison and complete-image endpoints | `run_v8_calibrated_final_factorial.py`; `analyze_v8_calibrated_final_factorial.py` | `v8_calibrated_final_factorial/` |
| Nominal RC thermal reconstruction | `run_v9_thermal_augmented_factorial.py`; `analyze_v9_thermal_augmented_factorial.py` | `v9_thermal_augmented_factorial/` |
| Main crossed-comparison figure | `plot_main_factorial_comparison.py` | `v9_restored_main_figures/` |
| Mandatory terminal outcomes and service map (Supplementary Figs. 10--11) | `run_v10_mandatory_outcome_audit.py`; `analyze_v10_mandatory_outcome_audit.py` | `v10_mandatory_outcome_audit/` |
| Unified overall comparison | `run_v11_unified_overall.py`; `analyze_v11_unified_overall.py` | `v11_unified_overall/` |
| Merged Overall figure with paper-aligned HEAT | `plot_v14_unified_overall_corrected_heat.py` | `v14_unified_overall_corrected_heat/` |
| Protected-mandatory certificate audit | `audit_v12_protected_certificate.py` | `v12_protected_certificate_audit/` |
| ESATD level sensitivity | `run_v13_esatd_unified_levels.py` | `v13_esatd_unified_levels/` |
| Full-image T4-profiled boundary | `run_v13_full_image_t4_boundary.py`; `analyze_v13_full_image_t4_boundary.py` | `v13_full_image_t4_boundary/` |
| HEAT non-preemptive DNN adaptation | `run_v13_heat_paper_aligned.py`; `analyze_v13_heat_paper_aligned.py` | `v13_heat_paper_aligned/` |
| Strict all-regions-mandatory matched grid | `run_v14_strict_all_regions_matched.py` | `v14_strict_all_regions_matched/` |
| HEAT 100-ms feasibility boundary | `run_v15_heat_100ms_feasible_domain.py` | `v15_heat_100ms_feasible_domain/` |
| Thermal-model and ambient sensitivity | `run_v17_thermal_deployment_sensitivity.py`; `analyze_v17_thermal_deployment_sensitivity.py` | `v17_thermal_deployment_sensitivity/` |
| Low-voltage branch mechanism | `run_v18_wide_cooling_mechanism.py`; `analyze_v18_wide_cooling_mechanism.py` | `v18_wide_cooling_mechanism/` |
| Top-$K_M$ service--perception trade-off | `run_v19_topk_mandatory_sweep.py`; `analyze_v19_topk_mandatory_sweep.py` | `v19_topk_mandatory_sweep/` |
| Multi-cue priority quality and charged cost | `run_v21_multicue_priority_aligned.py`; `analyze_v21_multicue_priority_aligned.py` | `v21_multicue_priority_aligned/` |
| Held-out calibration and aligned quality--cost figure (Supplementary Fig. 16) | `plot_v21_priority_calibration_quality_cost.py` | `v21_multicue_priority_aligned/` plus `perception_evidence/histogram_direction_calibration/` |
| Controlled priority-discrimination envelope | `run_v22_priority_discrimination_envelope.py`; `analyze_v22_priority_discrimination_envelope.py` | `v22_priority_discrimination_envelope/` |
| Hidden execution-time overrun robustness | `run_v4_r2_6_hidden_overrun.py`; `analyze_v4_r2_6_hidden_overrun.py` | `v4_r2_6_hidden_overrun/` |
| Published mean scheduler-overhead table | aggregate only; original repetition-level log was not retained | `scheduler_overhead/reported_mean_overhead.csv` and `scheduler_overhead/PROVENANCE.md` |

`plot_r2_priority_and_topk_evidence.py` recreates the load-stratified
Supplementary Figs. 19 and 20 from v19 and v22 while preserving the separate
aggregate Top-K trade-off used by Supplementary Fig. 14. Perception-only protocols and aggregate outputs are
under `perception_code/` and `perception_evidence/`, respectively.

The principal inference unit is the paired seed cluster where stated by the
corresponding analyzer. Checkpoint JSON files record grid dimensions, frozen
inputs, and protocol metadata. See `PROVENANCE.md` for the distinction among
the measured nominal-rate and power-capped T4 profiles, host-CPU measurements,
and RC-model thermal results.
