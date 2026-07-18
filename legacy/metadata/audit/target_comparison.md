# BRFSS 2024 target comparison

Same `core+ses+aces` feature set, 3-fold CV, simple LR baseline (impute + LR, no SMOTENC).

| target | prevalence | missingness | n_valid | baseline_roc_auc | baseline_f1 |
|---|---|---|---|---|---|
| ADDEPEV3 | 0.211 | 0.000 | 100,000 | 0.758 | 0.322 |
| _MENT14D | 0.601 | 0.017 | 98,284 | 0.760 | 0.772 |
| MENTHLTH_b14 | 0.347 | 0.608 | 39,181 | 0.723 | 0.473 |
| LSATISFY_dissat | 0.055 | 0.557 | 44,300 | 0.859 | 0.229 |


## Cross-tab vs ADDEPEV3 (current target)

Shows how much each candidate target overlaps with the current target. If overlap > 90%, switching target adds little new information.

### _MENT14D

(computed inline during evaluation)

### MENTHLTH_b14

(computed inline during evaluation)

### LSATISFY_dissat

(computed inline during evaluation)

## Cross-tabulations

### _MENT14D vs ADDEPEV3 (rows=ADDEPEV3, cols=_MENT14D)

```
_MENT14D      0      1    All
target                       
0         23193  54428  77621
1         15988   4675  20663
All       39181  59103  98284
```

### MENTHLTH_b14 vs ADDEPEV3 (rows=ADDEPEV3, cols=MENTHLTH_b14)

```
MENTHLTH      0      1    All
target                       
0         17745   5448  23193
1          7823   8165  15988
All       25568  13613  39181
```

### LSATISFY_dissat vs ADDEPEV3 (rows=ADDEPEV3, cols=LSATISFY_dissat)

```
LSATISFY      0     1    All
target                      
0         34005  1023  35028
1          7844  1428   9272
All       41849  2451  44300
```
