# Scheduling-overhead summary provenance

`reported_mean_overhead.csv` is the machine-readable form of the 25 mean
host-CPU scheduling times reported in Table 5 of the published article. It is
included so that the published table is represented in the artifact and can be
checked mechanically.

The original repetition-level benchmark log and the historical benchmark
driver were not retained in the final project archive. Consequently, this CSV
is a reported aggregate, not raw timing evidence, and it cannot be used to
recompute dispersion or tail latency. The article likewise treats these means
as prototype runtime characterization rather than a verified hard-real-time
execution-time bound.
