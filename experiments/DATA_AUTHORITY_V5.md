# Data authority V5

Frozen for the post-review rerun: 2026-08-15.

## Hardware provenance (mandatory reporting rule)

The current rerun and measurement host uses an **NVIDIA T4**.  Python
simulations, analysis scripts, and explicitly labelled prototype neural-network
benchmarks are executed on this host. Running a discrete-event simulation on
the T4 host does not turn a controlled simulation input into a hardware
measurement.

The GPU 0 execution-time vector below is measured in the nominal-rate T4
environment. The GPU 1 vector is also measured on T4 after applying the
declared slowdown/power-capped operating mode and is used to represent lower
service capacity. The labels denote operating profiles and do not imply that
two physical accelerators were measured simultaneously. Every table, caption,
checkpoint manifest, and response paragraph must distinguish these provenance
classes:

1. `T4_measured_input`: measurements supplied from the real T4 experiment;
2. `power_capped_T4_measured_input`: the measured slowed/power-capped T4
   service profile;
3. `T4_prototype_measurement`: GPU measurements made on the T4 experiment host
   under a separately stated prototype protocol.
4. `T4_host_CPU_measurement`: CPU-only measurements made on the T4 experiment
   host with the GPU excluded from the timed region.

Numerical conversion between different measurement protocols is not an
authoritative T4 measurement and must not be used in the revised paper. Such
values may appear only as explicitly labelled exploratory sensitivity inputs.

The multi-cue priority scorer is timed after image preload and uses CPU feature
extraction and classification. These timings are therefore CPU-side
measurements obtained on the T4 experiment host, not T4 GPU inference times.
They may be reported as measured host preprocessing overhead and charged to the
scheduling experiment when the stated implementation and protocol apply.

## Execution-time and scheduling-quality inputs

- GPU 0 T4-measured execution-time budget for L1--L5: 6.3, 9.0, 12.3,
  14.8, 20.1 ms (`T4_measured_input`).
- GPU 1 measured power-capped/slowed T4 service profile for L1--L5: 13.0,
  18.2, 25.1, 29.5, 40.0 ms (`power_capped_T4_measured_input`).
- Historical image-averaged path Dice for L1--L5: 0.5790, 0.6183,
  0.6441, 0.6730, 0.6884.

The historical image-averaged values are the only scalar path-quality inputs
used by Stage-I optimization and by the historical scheduling-utility audit.
They must not be described as pooled pixel micro-Dice.

## End-to-end perception metrics

`D_CI`, pixel recall, image miss, and completed-only Dice are recomputed from
the frozen per-image, per-region, per-level confusion counts. Skipped and late
regions receive zero prediction credit and their ground-truth positive pixels
are counted as false negatives. These metrics are outputs, not scheduling
inputs.

The separately observed pooled pixel micro-Dice values must not replace the
historical image-averaged scheduling-quality vector and must not be used to
claim that L3 is the most accurate path under the historical Average Dice
definition. If reported, they require an explicit `micro` label and a separate
aggregation explanation.

## Algorithm semantics

Stage II uses the Stage-I selected-level budget,
`mu = C(processor, selected_level, 0.8 V)`, matching the original executable
statement `mu.append(task.ultilization[j][l])`. Mandatory work is ordered before
optional work when absolute deadlines tie. Actual execution overruns, when
enabled only for R2.6 robustness analysis, are sampled after dispatch.

## Superseded evidence

`v4_final_factorial` predates the correction that assigns the selected-level
budget at Stage-I construction time. Its checkpoint is retained for audit but
must not be cited as final factorial evidence. The authoritative rerun is
`v5_final_factorial_selected_budget`.
