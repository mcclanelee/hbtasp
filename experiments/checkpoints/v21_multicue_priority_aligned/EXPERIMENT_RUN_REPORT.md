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

All 600 reported cells use the frozen held-out calibrated histogram pool. The
final package contains only this validated experiment grid; terminal accounting
is exact and no admitted mandatory region finishes after its deadline.
