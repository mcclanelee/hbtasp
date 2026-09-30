# Productive-cooling same-kernel ablation

All 560 paired cells are present, terminal accounting is exact, and nominal
thermal violations are zero. The control differs only by disabling the
productive-cooling search branch.

Cooling-minus-control differences are
+0.000000 for mandatory
rejection, -0.002364 for
complete-image Dice,
-0.000148 degrees C for
peak modeled temperature, and
-0.004677 V for mean selected voltage.
These paired effects, including their confidence intervals in
`paired_seed_contrasts.csv`, define the evidential scope of the cooling claim.
The confidence intervals use the ten seeds as independent inferential units;
`paired_contrasts.csv` is retained only as a cell-level sensitivity summary.

At 200 ms, cooling-minus-control differences are
+0.000000 for
mandatory rejection,
+0.000000
for admitted mandatory deadline failure,
+0.00001125 degrees C
for peak modeled temperature, and
-0.004087 V for mean selected
voltage. Thus aggregation across periods does not conceal a service-level or
thermal benefit at the historical 200-ms operating point.
