# Top-K Mandatory-Region Sweep: Result Analysis

## Protocol

- Frozen calibrated 500-image replay pool.
- HBTASP Dynamic L1--L5 scheduling kernel.
- Mandatory cardinality `K_M` in `{1, 2, 3, 4}`, selected by the same calibrated priority ordering.
- Periods: 100, 150, 200, 250, and 300 ms.
- Production lines: 4, 6, 8, and 10.
- Ten paired seeds and 1,000 release cycles per cell.
- 800 complete cells; mandatory terminal residual is zero in every cell.

## Grand means

| `K_M` | Mandatory service failure | Defect-positive regions marked mandatory | Defect-positive regions returned on time | Images with all defect-positive regions on time | `D_CI` | Recall | Image complete-miss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0325 | 0.3137 | 0.8154 | 0.7390 | 0.4068 | 0.5302 | 0.1688 |
| 2 | 0.1821 | 0.5664 | 0.7030 | 0.5985 | 0.3637 | 0.4548 | 0.2648 |
| 3 | 0.3413 | 0.7882 | 0.6273 | 0.5523 | 0.3167 | 0.3982 | 0.3636 |
| 4 | 0.4684 | 1.0000 | 0.5268 | 0.4980 | 0.2667 | 0.3330 | 0.4820 |

`K_M=4` has no optional regions, so optional coverage is not applicable.

## Interpretation

Increasing `K_M` makes a larger share of defect-positive regions mandatory, but the resulting
pre-execution service loss reduces the share actually returned before the deadline. Relative to
`K_M=1`, `K_M=2` raises mandatory service failure by 0.1496 and lowers defect-positive-region
on-time coverage by 0.1124, complete-image Dice by 0.0431, and recall by 0.0754. The losses grow
at `K_M=3` and `K_M=4`.

Within the tested two-processor, 100--300-ms, 4--10-line grid, `K_M=1` therefore dominates the
larger mandatory sets on mandatory service failure, defect-positive-region on-time coverage,
complete-image Dice, recall, image complete-miss, and full defect-region image coverage. The
result does not establish global optimality outside this operating envelope. It shows that, in
the stated target envelope, nominally protecting more regions consumes enough admission capacity
to reduce rather than improve delivered defect coverage.

## Regression gates

- `K_M=1` matches all 200 official HBTASP--Dynamic cells exactly for mandatory service failure,
  `D_CI`, recall, and image complete-miss (maximum absolute difference 0).
- `K_M=4` matches all 200 official all-regions-mandatory cells exactly for the same metrics
  (maximum absolute difference 0).
- The calibrated pool contains 500 images, and its stored `mandatory_idx` agrees with the
  recalculated Top-1 priority for every image.

## Claim boundary

The experiment supports `K_M=1` as the empirically best delivered-service configuration among
the four tested cardinalities in this workload envelope. It does not show that a single mandatory
region guarantees image-level recall, nor that it is optimal for other processors, looser loads,
different priority estimators, or applications with a strict complete-coverage requirement.

