# Execution-time-overrun robustness analysis

The scheduler is given only the frozen measured WCET. An overrun is sampled
after dispatch and is intentionally withheld from the scheduler when it selects
the GPU, DNN level, and voltage.
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

## Robustness interpretation

The experiment verifies zero nominal violations under the measured-WCET
contract and extends the evaluation with rare model-external execution-time
overruns. The resulting small transient excursions explicitly characterize the
robustness boundary beyond nominal conditions. The illustrative 10-second
trace uses three controlled post-dispatch injections near 2.5, 6, and 8
seconds; the paired grid above provides the quantitative event-rate analysis.
