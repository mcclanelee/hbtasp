# HBTASP reproducibility package

This anonymous package contains the scheduling/event-replay implementation,
analysis scripts, cell-level results, and perception-evidence summaries used in
the revised manuscript.

## Evidence tracks

- Regional and full-image GPU execution budgets are empirical NVIDIA T4
  profiles under the protocols recorded in the checkpoint metadata.
- The second processor is an explicitly degraded service profile used to model
  heterogeneous capacity.
- Temperature results are first-order RC-model simulations driven by the
  execution profiles.
- Priority-estimator overhead is a prototype host-CPU preprocessing measurement
  with images preloaded and is kept separate from the T4 GPU profiles.

## Reproduction

1. Create a clean Python environment.
2. Install `requirements.txt`; perception reruns additionally use
   `requirements-perception.txt`.
3. Run `python verify_release.py` to validate the package structure, hashes,
   principal grids, and anonymous-release rules.
4. Run the scripts in `experiments/` from the repository root. Each script
   records its protocol and output location in the corresponding checkpoint.

The Severstal images and trained perception weights are not redistributed.
Perception reruns require the public dataset and model training described by the
included scripts.
