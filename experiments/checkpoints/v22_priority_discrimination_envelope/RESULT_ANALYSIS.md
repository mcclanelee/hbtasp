# Priority-discrimination envelope on the principal grid

All 600 cells are present and paired over five periods, four production-line
counts, and ten seeds. Terminal accounting is exact, and no admitted mandatory
region finishes late. The calibrated-histogram cells are the validated v21
cells; frozen-random and ground-truth-informed diagnostic cells use the same
HBTASP kernel and paired grid.

Priority diagnostics on the frozen 500-image pool:

- calibrated histogram: hit rate 62.80%, defect-pixel share 34.85%;
- frozen random ranking: hit rate 51.60%, defect-pixel share 25.46%;
- ground-truth-informed diagnostic reference: hit rate 100.00%, defect-pixel share 76.39%.

Grand means (seed is the inference unit):

```
metric                image_complete_miss_rate  mandatory_service_failure  mean_complete_image_dice  pixel_defect_recall
strategy                                                                                                                
calibrated_histogram                  0.168846                     0.0325                  0.406796             0.530182
random_frozen                         0.197233                     0.0325                  0.386921             0.511026
defect_oracle                         0.091088                     0.0325                  0.496683             0.594609
```

Relative to the calibrated histogram, frozen random ranking changes
complete-image Dice by -0.0199 and pixel recall by -0.0192.
The ground-truth-informed diagnostic reference changes complete-image Dice by +0.0899 and recall by
+0.0644. Because all policies protect exactly one region and add
zero priority-computation latency, mandatory-service failure mainly measures
capacity, while Dice, recall, and image miss expose the delivered-perception
effect of ranking quality. The diagnostic-reference gap quantifies the
remaining improvement opportunity for lightweight deployable estimators.
