# Result provenance audit

- Raw cells: `cell_results.csv` (600 rows).
- Treatments: histogram_zero, multicue_zero, multicue_host_cpu_p99.
- Grid: 5 periods x 4 line counts x 10 seeds x 3 treatments.
- Maximum terminal-accounting residual: 0.0.
- Admitted mandatory deadline violations: 0.
- Priority timing: `priority_latency_4_6_8_10.json`, independently measured
  on the CPU of the T4 experiment host with images preloaded.
- Aggregation: `experiments/analyze_v21_multicue_priority_aligned.py`.

Verdict: FULLY_TRACEABLE_AND_CONSISTENT.
