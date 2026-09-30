# HEAT paper contract for the DNN adaptation

Source: Y. Sharma and S. Moulik, *HEAT: A heterogeneous multicore
real-time scheduler with efficient energy and temperature management*, ACM
SIGAPP Applied Computing Review 22(2), 2022, Algorithms 1--5.

## Binding algorithm rules

1. HEAT constructs deadline-partitioned time slices over a periodic task-set
   hyperperiod.  In the present implicit-period grid, one period is one repeated
   time slice.
2. `CLUSTER-DESIGN` forms clusters containing at most two cores and rejects a
   task set if every task cannot be assigned within cluster capacity.
3. `SCHEDULE-DESIGN` orders tasks by their cross-core utilization ratio, fills
   the preferred core, then the second core, and permits at most one migrating
   task per two-core cluster and time slice.
4. `TEMP-DESIGN` classifies non-migrating tasks using their *offline profiled
   task-specific steady-state temperatures* and alternates cool and hot tasks.
   It does not implement a current-temperature feedback rule and does not raise
   voltage when a core exceeds an online threshold.
5. `ENERGY-DESIGN` calculates
   `Fopt = sum(fixed shares)/(slice length - sum(migrating shares))` and rescales
   non-migrating shares.  Frequency and voltage are treated as linearly related.
6. HEAT reports task-set success/acceptance.  An infeasible task set is rejected;
   it is not executed indefinitely as a backlogged queue.

## Explicit DNN adaptation boundary

- A single DNN inference is non-preemptive and cannot migrate midway between
  the two GPUs.  The paper's fluid/migrating-task operation is therefore
  disabled rather than simulated with an unsupported split inference.
- At one fixed CS-DNN level, every region has the same profiled service and RC
  steady-state model.  Consequently, the paper's hot/cold task ordering is a
  declared no-op for this workload; no artificial per-image thermal diversity
  is introduced.
- The adapter preserves cluster-capacity rejection, preferred-core ordering,
  and Eq. (4) frequency selection.  It is named `HEAT non-preemptive DNN
  adaptation`, not a bit-for-bit reproduction of the paper's fluid scheduler.
- ATP/IIT are reported only for accepted task-set cells.  Rejected cells have no
  execution trajectory and hence no thermal-performance value.
