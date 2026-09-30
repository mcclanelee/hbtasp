# V9 thermal-augmented calibrated factorial

All 800 cells are present. Mandatory DMR and complete-image Dice are
identical cell by cell to the authoritative V8 factorial. The continuous
RC reconstruction has zero nominal IIT, zero thermal violations, and no
peak above 60 C. ATP is therefore an additional descriptive trajectory
metric, not a rerun with altered scheduling decisions.

## Overall

```
          configuration  mandatory_dmr  mean_complete_image_dice  average_temperature_c  iit_celsius_seconds  peak_temperature_c
EDF-Dynamic-Reservation       0.212706                  0.398007              58.620852                  0.0           59.490469
            EDF-FixedL3       0.259452                  0.397304              53.864582                  0.0           55.227818
         HBTASP-Dynamic       0.032500                  0.406796              59.307122                  0.0           59.999861
         HBTASP-FixedL3       0.000000                  0.449802              56.996140                  0.0           59.794063
```

## By period

```
          configuration  period_ms  mandatory_dmr  mean_complete_image_dice  average_temperature_c  iit_celsius_seconds  peak_temperature_c
EDF-Dynamic-Reservation        100       0.500358                  0.265989              58.600569                  0.0           59.634908
EDF-Dynamic-Reservation        150       0.312487                  0.360765              58.968832                  0.0           59.782487
EDF-Dynamic-Reservation        200       0.151777                  0.425316              59.356300                  0.0           59.880005
EDF-Dynamic-Reservation        250       0.075714                  0.460691              58.591833                  0.0           59.622077
EDF-Dynamic-Reservation        300       0.023195                  0.477275              57.586725                  0.0           58.532867
            EDF-FixedL3        100       0.588508                  0.220782              53.634558                  0.0           55.920073
            EDF-FixedL3        150       0.358333                  0.344455              55.671848                  0.0           57.159399
            EDF-FixedL3        200       0.207917                  0.424994              54.854859                  0.0           55.992537
            EDF-FixedL3        250       0.106250                  0.479159              53.439231                  0.0           54.353046
            EDF-FixedL3        300       0.036250                  0.517132              51.722413                  0.0           52.714035
         HBTASP-Dynamic        100       0.162500                  0.272953              59.536973                  0.0           59.999881
         HBTASP-Dynamic        150       0.000000                  0.368912              59.627377                  0.0           59.999961
         HBTASP-Dynamic        200       0.000000                  0.438275              59.632170                  0.0           59.999964
         HBTASP-Dynamic        250       0.000000                  0.469731              59.348922                  0.0           59.999644
         HBTASP-Dynamic        300       0.000000                  0.484107              58.390169                  0.0           59.999855
         HBTASP-FixedL3        100       0.000000                  0.321520              59.528289                  0.0           59.999953
         HBTASP-FixedL3        150       0.000000                  0.420715              59.449253                  0.0           59.999858
         HBTASP-FixedL3        200       0.000000                  0.477479              56.695540                  0.0           59.999962
         HBTASP-FixedL3        250       0.000000                  0.504356              56.378890                  0.0           59.999747
         HBTASP-FixedL3        300       0.000000                  0.524938              52.928729                  0.0           58.970796
```