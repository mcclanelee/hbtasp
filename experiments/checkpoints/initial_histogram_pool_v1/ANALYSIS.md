# Uncalibrated histogram reference pool

This checkpoint joins the reference label-free symmetric
chi-square score with the final-test mask replay records by `image_id` and
region index. The intersection contains 500 images (2,000 regions). The larger
histogram audit contains 1,334 images, but only the 500-image intersection has
the mask/confusion information required for end-to-end scoring, so the two
sample sizes are reported separately.

The background templates were constructed from label-confirmed clean training
regions. Test labels are retained only for subsequent evaluation and are not
used to compute or orient scheduler weights. Scores are normalized within each
image and the maximum score is selected as mandatory under the reference
protocol. Preserving this original score direction provides a controlled
reference for the held-out direction-calibration experiment.

Pool diagnostics:

- all 500 images have exactly four histogram scores;
- every image's normalized weights sum to one;
- the selected mandatory region is defective in 32.8% of images;
- 62.0% of images contain defects in more than one region;
- the full 1,334-image histogram audit reports ROC-AUC 0.35178 for the original
  score direction.

These diagnostics motivate the held-out direction calibration used by the
final priority estimator. This pool is retained as an uncalibrated reference,
whereas the final priority analyses use calibrated ranking and deadline-aware
metrics that assign zero credit to skipped or late defective regions.
