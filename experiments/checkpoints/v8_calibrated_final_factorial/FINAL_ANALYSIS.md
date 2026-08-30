# Final 2x2 factorial analysis

Status: scheduling, historical Average-Dice utility, and zero-credit confusion-replay evidence.

All 800 cells are present (200/configuration), terminal accounting residual is zero, and modeled thermal violations are zero.

## Same-model primary means

| Configuration | Mandatory DMR | Completed-only Dice | Coverage-adjusted Dice | Weighted coverage utility | Mandatory effective Dice |
|---|---:|---:|---:|---:|---:|
| EDF-Dynamic-Reservation | 0.2127 | 0.6344 | 0.5346 | 0.5366 | 0.5420 |
| EDF-FixedL3 | 0.2595 | 0.6441 | 0.4750 | 0.4753 | 0.4770 |
| HBTASP-Dynamic | 0.0325 | 0.6424 | 0.5087 | 0.5528 | 0.6660 |
| HBTASP-FixedL3 | 0.0000 | 0.6441 | 0.4919 | 0.5390 | 0.6441 |

## HBTASP-Dynamic minus EDF-Dynamic-Reservation

| Metric | Difference | 95% CI | p | Wins/200 |
|---|---:|---:|---:|---:|
| mandatory_dmr | -0.18021 | [-0.20386, -0.15655] | 1.35e-34 | 0 |
| historical_completed_only_dice | 0.00792 | [0.00568, 0.01016] | 4.32e-11 | 118 |
| historical_coverage_adjusted_dice | -0.02586 | [-0.03048, -0.02124] | 2.13e-22 | 16 |
| historical_weighted_coverage_utility | 0.01621 | [0.01349, 0.01893] | 1.41e-24 | 150 |
| historical_mandatory_effective_dice | 0.12405 | [0.10777, 0.14034] | 1.35e-34 | 160 |
| mean_complete_image_dice | 0.00879 | [0.00735, 0.01023] | 2.42e-25 | 155 |
| pixel_defect_recall | -0.01370 | [-0.01882, -0.00858] | 3.38e-07 | 83 |
| image_complete_miss_rate | -0.02699 | [-0.03270, -0.02129] | 2.14e-17 | 65 |

Weighted utility is the manuscript-objective metric. Coverage-adjusted Dice is unweighted and is retained for R2.8 transparency. No universal dominance claim is made.
