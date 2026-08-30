# Initial histogram replay pool V1

This checkpoint joins the initial manuscript's label-free symmetric
chi-square score with the final-test mask replay records by `image_id` and
region index. The intersection contains 500 images (2,000 regions). The larger
histogram audit contains 1,334 images, but only the 500-image intersection has
the mask/confusion information required for R2.8 end-to-end scoring; these
sample sizes must not be conflated.

The background templates were constructed from label-confirmed clean training
regions. Test labels are retained only for subsequent evaluation and are not
used to compute or orient scheduler weights. Scores are normalized within each
image and the maximum score is selected as mandatory exactly as in the initial
manuscript. The score direction is not reversed despite unfavorable evidence.

Pool diagnostics:

- all 500 images have exactly four histogram scores;
- every image's normalized weights sum to one;
- the selected mandatory region is defective in 32.8% of images;
- 62.0% of images contain defects in more than one region;
- the full 1,334-image histogram audit reports ROC-AUC 0.35178 for the original
  score direction.

Therefore this pool is eligible as a faithful input, but it already confirms
that the histogram score cannot be described as a calibrated defect
probability. It also cannot support a claim that executing only the maximum-
weight region guarantees industrial recall. All skipped and late defective
regions must be penalized by the R2.8 metric replay.
