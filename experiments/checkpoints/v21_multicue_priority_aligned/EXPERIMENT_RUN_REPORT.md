# Experiment run report

The aligned Top-1 priority experiment completed 600/600 cells:

- treatments: calibrated histogram, multi-cue with no charged latency, and
  multi-cue with independently measured host-CPU p99 latency;
- periods: 100, 150, 200, 250, and 300 ms;
- production-line counts: 4, 6, 8, and 10;
- seeds: 101, 202, 303, 404, 505, 606, 707, 808, 909, and 1010;
- replay length: 1,000 release cycles per cell.

The multi-cue latency was independently remeasured 500 times for every line
count with images preloaded. The p99 values charged to the event replay were
24.884, 33.311, 53.682, and 67.341 ms for 4, 6, 8, and 10 lines,
respectively. These are CPU-side measurements obtained on the T4 experiment
host; the T4 GPU is excluded from the timed region.

The initial 72-cell attempt used an obsolete histogram pool. It was stopped,
excluded, and preserved under `discarded_initial_pool_run/`. Every reported
result uses the held-out calibrated histogram pool used by the current paper.
