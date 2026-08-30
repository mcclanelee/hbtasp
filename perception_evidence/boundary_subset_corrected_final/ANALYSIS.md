# Dedicated boundary-defect subset

The subset was fixed from ground truth before predictions were inspected and
contains 468 images and 641 class-specific connected components crossing an
original cut at x=400, 800, or 1200. Confidence intervals use 10,000 paired
cluster bootstrap resamples at image level.

Heavy overlap versus non-overlap changes ROI Dice by
+0.00524 (95% CI
-0.00048 to
+0.01105), pixel recall by
+0.00647, and IoU>=0.1 component recall
by +0.03276. Boundary-image complete
miss rate changes by
-0.02137. These benefits
must be weighed against the separately measured prototype T4 p99 processing-time increase;
they do not establish universal superiority.
