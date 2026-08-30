# Aligned Top-1 priority quality--latency experiment

All 600 cells are present and exactly paired over periods 100--300 ms, line
counts 4/6/8/10, and ten seeds. Terminal accounting is exact and no admitted
mandatory-region deadline violation occurs.

Grand means:

```
                       mandatory_service_failure  mean_complete_image_dice  pixel_defect_recall  image_complete_miss_rate
treatment                                                                                                                
histogram_zero                          0.032500                  0.406796             0.530182                  0.168846
multicue_zero                           0.032500                  0.446639             0.563138                  0.116538
multicue_host_cpu_p99                   0.130417                  0.389710             0.476737                  0.219143
```

The zero-latency treatment isolates the change in priority ordering. The host-
CPU-p99 treatment then charges the independently measured feature-extraction
and classifier latency. This timing is a host-specific sensitivity input and
must not be described as a T4 timing measurement.
