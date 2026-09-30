# Audited 2x2 scheduler--network effect decomposition

Source: `experiments\checkpoints\v8_calibrated_final_factorial\cell_results.csv`

The analysis uses all 800 audited cells (200 per configuration). The ten paired seeds are the independent inferential units; the five periods and four line counts are averaged within each seed before forming Student-t 95% confidence intervals.

A negative difference favors the first-named configuration for mandatory service failure and image complete-miss rate. A positive difference favors it for complete-image Dice and recall.

| Effect | Metric | Mean difference | Seed-clustered 95% CI |
|---|---|---:|---:|
| Scheduling effect under Fixed L3 | Mandatory service failure | -0.2595 | [-0.2599, -0.2590] |
| Scheduling effect under Fixed L3 | Complete-image Dice | +0.0525 | [+0.0516, +0.0534] |
| Scheduling effect under Fixed L3 | End-to-end recall | +0.0407 | [+0.0380, +0.0434] |
| Scheduling effect under Fixed L3 | Image complete-miss rate | -0.1328 | [-0.1338, -0.1318] |
| Scheduling effect under Dynamic L1--L5 | Mandatory service failure | -0.1802 | [-0.1824, -0.1780] |
| Scheduling effect under Dynamic L1--L5 | Complete-image Dice | +0.0088 | [+0.0077, +0.0098] |
| Scheduling effect under Dynamic L1--L5 | End-to-end recall | -0.0137 | [-0.0174, -0.0100] |
| Scheduling effect under Dynamic L1--L5 | Image complete-miss rate | -0.0270 | [-0.0280, -0.0260] |
| Dynamic-path effect under EDF | Mandatory service failure | -0.0467 | [-0.0491, -0.0444] |
| Dynamic-path effect under EDF | Complete-image Dice | +0.0007 | [-0.0007, +0.0021] |
| Dynamic-path effect under EDF | End-to-end recall | +0.0636 | [+0.0583, +0.0690] |
| Dynamic-path effect under EDF | Image complete-miss rate | -0.0981 | [-0.0997, -0.0966] |
| Dynamic-path effect under HBTASP | Mandatory service failure | +0.0325 | [+0.0325, +0.0325] |
| Dynamic-path effect under HBTASP | Complete-image Dice | -0.0430 | [-0.0437, -0.0423] |
| Dynamic-path effect under HBTASP | End-to-end recall | +0.0092 | [+0.0079, +0.0106] |
| Dynamic-path effect under HBTASP | Image complete-miss rate | +0.0077 | [+0.0070, +0.0084] |
| Scheduler--path interaction | Mandatory service failure | +0.0792 | [+0.0769, +0.0816] |
| Scheduler--path interaction | Complete-image Dice | -0.0437 | [-0.0453, -0.0421] |
| Scheduler--path interaction | End-to-end recall | -0.0544 | [-0.0602, -0.0486] |
| Scheduler--path interaction | Image complete-miss rate | +0.1058 | [+0.1041, +0.1075] |

## Interpretation boundary

The crossed design separates the effect of access to Dynamic L1--L5 paths from the effect of replacing the EDF reservation control with the integrated HBTASP scheduling policy. It does not by itself identify the contribution of each internal HBTASP component; Static-V and the existing cooling ablations serve that separate purpose.
