# HBTASP reproducibility package

This repository contains the executable scheduling and event-replay
code, frozen cell-level results, analysis scripts, and perception summaries for
the article *HBTASP: Two-Stage Temperature-Aware Multiprocessor Scheduling for
Dynamic DNNs in Real-Time Defect Detection*, published online in *Expert Systems
with Applications* (Article 134539; <https://doi.org/10.1016/j.eswa.2026.134539>),
and its accompanying Supplementary Material. The anonymous repository snapshot
cited in the published article remains available at
<https://anonymous.4open.science/r/hbtasp-62C1/>.

## What the evidence represents

- GPU 0 uses the measured nominal-rate NVIDIA T4 execution-time vector. GPU 1
  uses the measured power-capped (deliberately slowed) T4 service vector. The
  two vectors are empirical profiles of the two operating modes used to model
  heterogeneous service capacity; they do not require two simultaneously
  instrumented physical accelerators.
- Temperature, IIT, and thermal-feasibility results are first-order RC-model
  simulations parameterized for the profiled T4 operating regime. They are not
  synchronized physical T4 temperature measurements.
- Full-image and regional model timings are identified by their checkpoint
  protocols. Priority-estimator cost is a host-CPU preprocessing measurement
  with images preloaded and is separate from GPU inference time.
- Rejected, skipped, or late regions receive zero prediction credit in the
  reported complete-image Dice, pixel defect recall, and image complete-miss
  metrics.

These boundaries match `PROVENANCE.md`, `experiments/DATA_AUTHORITY_V5.md`,
and the published article's experimental-setting and limitation statements.

## Quick verification

The paper's reference environment is Ubuntu 20.04 with Python 3.10, PyTorch
2.1.2 (CUDA 12.1), and SciPy 1.9.1. The scheduling and analysis checks are also
compatible with Python 3.8 or later under the dependency ranges below.

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python verify_manifest.py
python verify_release.py
```

`verify_manifest.py` is read-only and checks every packaged file against
`RELEASE_SHA256.json`. `verify_release.py` is also read-only and checks row
counts, terminal-state conservation, nominal thermal invariants, and the final
Top-K and priority-discrimination protocols. `generate_manifest.py` is a
maintainer-only command to run after an intentional release change.

## Repository layout

- `experiments/`: schedulers, event-replay runners, analyzers, protocol locks,
  and frozen checkpoints.
- `models/`: reference ESATD and HEAT implementations. Unified comparisons use
  the explicitly documented adapters under `experiments/`.
- `perception_code/`: model training/evaluation and priority-estimator code.
- `perception_evidence/`: frozen aggregate perception results.
- `mask_replay_final_test_shared/`: frozen region/level confusion counts used
  for deadline-aware replay.
- `docs/REPRODUCIBILITY_MAP.md`: published-article claim-to-code/result map.

The Severstal images and trained neural-network weights are not redistributed.
The protocol uses 6,666 defect-positive source images split by source identifier
into 4,666 training, 666 calibration, and 1,334 test images; the first 500 test
identifiers form the frozen scheduling pool. Download the public dataset and
install `requirements-perception.txt` in addition to the base requirements for
model-level reruns. Perception utilities support module execution from the
repository root as well as direct execution from `perception_code/`.

## Principal experiment entry points

Run experiment modules from the repository root. The frozen checkpoints allow
the analyzers and verification commands to run without repeating long grids.

```bash
python -m experiments.run_v8_calibrated_final_factorial
python -m experiments.analyze_v8_calibrated_final_factorial
python -m experiments.run_v9_thermal_augmented_factorial
python -m experiments.run_v11_unified_overall
python -m experiments.run_v19_topk_mandatory_sweep
python -m experiments.analyze_v19_topk_mandatory_sweep
python -m experiments.run_v21_multicue_priority_aligned
python -m experiments.analyze_v21_multicue_priority_aligned
python -m experiments.run_v22_priority_discrimination_envelope
python -m experiments.analyze_v22_priority_discrimination_envelope
```

The final service--perception evidence has three distinct roles:

- `v19_topk_mandatory_sweep` is the 800-cell Top-$K_M$ sweep for
  $K_M=1,\ldots,4$. It exposes the service--recall trade-off rather than
  asserting that Top-1 is universally preferable.
- `v21_multicue_priority_aligned` compares the calibrated histogram and a
  multi-cue estimator both before and after charging measured host-CPU
  preprocessing latency.
- `v22_priority_discrimination_envelope` is a 600-cell controlled replay at
  fixed Top-1 cardinality and zero ranking latency. It compares frozen random,
  calibrated-histogram, and ground-truth-informed diagnostic rankings. The
  `defect_oracle` identifier is an analysis-only upper reference, not a
  deployable policy. Because capacity and cardinality are fixed, equal average
  mandatory-region service failure in this grid does not imply equal delivered
  perception quality; the experiment isolates which region receives protected
  service. The charged-latency effect is assessed separately in v21.

To recreate the two load-stratified supplementary figures from the frozen v19
and v22 summaries, run:

```bash
python -m experiments.plot_r2_priority_and_topk_evidence
```

## Interpretation limits

The released results support conditional claims within the tested periods,
line counts, priority pools, latency profiles, and RC-model parameters. Mean
scheduler overhead is not a hard tail-latency bound. Likewise, Top-1 is a
low-cardinality graceful-degradation setting: it preserves protected service
under overload at the cost of leaving lower-ranked defective regions optional.
The calibrated histogram improves over random ranking but has limited
discrimination, so its errors can reduce complete-image Dice and pixel recall,
especially at tighter loads. A more accurate estimator is useful only when its
processing cost is included in admission and schedulability analysis.
