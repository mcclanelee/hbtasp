# R2.6 hidden-overrun audit

The scheduler is given only the frozen measured WCET. An overrun is sampled
after dispatch, so it cannot influence the selected GPU, DNN level, or voltage.
The primary paired grid contains 5 periods x 10 seeds at four lines and 1000
cycles per cell. The robustness condition uses probability 0.005 and a
conditional extra-duration factor U(0.10, 0.70).

## Main result

All 50 nominal cells have zero thermal violations, zero IIT, and
zero deadline misses. The observed hidden-overrun rate is
0.004832. Under hidden overruns, the mean peak
temperature is 60.086954 C, mean IIT is
0.002312254 C s per 1000-cycle cell, and mean
mandatory DMR is 0.000875. Positive IIT occurs in
48/50 cells.

## Interpretation boundary

The deterministic guarantee is empirically preserved when actual execution
does not exceed the measured WCET. Rare model-external execution overruns can
produce small transient excursions; therefore the manuscript must replace an
unqualified real-system claim of strict satisfaction with a conditional
guarantee plus robustness disclosure. The illustrative 10-second trace uses
three controlled post-dispatch injections near 2.5, 6, and 8 seconds; it is not
used to estimate the formal event rate.
