# Result Provenance Audit

Verdict: `FULLY_TRACEABLE_AND_CONSISTENT`

## Inputs

- Priority/replay pool: `experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json`
  - SHA-256: `31fe3ef9e44c2a3b9f8af9331f2727651cd3835a0c1f293097a3c754cae54b00`
- Mask confusion replay: `mask_replay_final_test_shared/mask_confusion_by_level.csv`
  - SHA-256: `83a89acd926bcaf895060407dac3fb2ed5e78e3ad009291ff9eee4d1e309ae50`

## Code

- Event kernel: `experiments/initial_manuscript_event_replay.py`
- Runner: `experiments/run_v19_topk_mandatory_sweep.py`
- Aggregation and figure: `experiments/analyze_v19_topk_mandatory_sweep.py`

## Outputs

- Cell-level results: `cell_results.csv` (800 unique cells).
- Protocol and hashes: `checkpoint.json`.
- Seed-level means and 95% intervals: `topk_summary_with_seed_ci.csv`.
- Period-level means: `topk_period_summary.csv`.
- Figure: `v19_topk_mandatory_tradeoff.pdf` and `.png`.

## Integrity checks

- Expected cells: `4 mandatory counts x 5 periods x 4 line counts x 10 seeds = 800`.
- Observed cells: 800.
- Duplicate keys: 0.
- Maximum mandatory terminal residual: 0.
- Top-1 regression against the official calibrated HBTASP--Dynamic run: exact in all 200 cells.
- Top-4 regression against the official all-regions-mandatory run: exact in all 200 cells.
- No manuscript or Response Letter file was modified by this experiment.

