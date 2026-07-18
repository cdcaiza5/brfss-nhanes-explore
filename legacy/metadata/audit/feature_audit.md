# BRFSS 2024 feature audit

## Tier counts

| tier | n |
|---|---|
| strong | 45 |
| moderate | 52 |
| weak | 150 |
| leakage | 5 |
| skip | 48 |

## Top-30 by RandomForest importance

| rank | name | tier | score |
|---|---|---|---|
| 1 | _MENT14D | leakage | 0.1541 |
| 2 | DECIDE | strong | 0.0881 |
| 3 | MENTHLTH | leakage | 0.0646 |
| 4 | POORHLTH | leakage | 0.0545 |
| 5 | DIFFALON | strong | 0.0259 |
| 6 | _PHYS14D | leakage | 0.0206 |
| 7 | EMPLOY1 | moderate | 0.0175 |
| 8 | GENHLTH | strong | 0.0170 |
| 9 | SDLONELY | strong | 0.0164 |
| 10 | _AGE80 | weak | 0.0143 |
| 11 | _RFHLTH | leakage | 0.0129 |
| 12 | _SEX | weak | 0.0125 |
| 13 | _BMI5 | weak | 0.0116 |
| 14 | PHYSHLTH | strong | 0.0113 |
| 15 | _AGEG5YR | weak | 0.0105 |
| 16 | ECIGNOW3 | moderate | 0.0094 |
| 17 | LSATISFY | strong | 0.0089 |
| 18 | ASTHMA3 | strong | 0.0088 |
| 19 | _LTASTH1 | weak | 0.0088 |
| 20 | _ASTHMS1 | weak | 0.0087 |
| 21 | HEIGHT3 | moderate | 0.0084 |
| 22 | _AIDTST4 | weak | 0.0083 |
| 23 | WTKG3 | weak | 0.0080 |
| 24 | _CASTHM1 | weak | 0.0078 |
| 25 | HAVARTH4 | strong | 0.0076 |
| 26 | _DRDXAR2 | weak | 0.0074 |
| 27 | HIVTST7 | moderate | 0.0074 |
| 28 | HIVTSTD3 | weak | 0.0073 |
| 29 | WEIGHT2 | moderate | 0.0072 |
| 30 | _AGE_G | weak | 0.0070 |

## Top-30 by Univariate (point-biserial | Cramér's V)

| rank | name | tier | score |
|---|---|---|---|
| 1 | _MENT14D | leakage | 0.4508 |
| 2 | DECIDE | strong | 0.3405 |
| 3 | MENTHLTH | leakage | 0.3275 |
| 4 | SDLONELY | strong | 0.3136 |
| 5 | CDSOCIA1 | strong | 0.2984 |
| 6 | CSRVCTL2 | weak | 0.2891 |
| 7 | ACEDEPRS | strong | 0.2827 |
| 8 | CDHOUS1 | strong | 0.2781 |
| 9 | LSATISFY | strong | 0.2777 |
| 10 | CIMEMLO1 | strong | 0.2556 |
| 11 | _BMI5 | weak | 0.2427 |
| 12 | _PHYS14D | leakage | 0.2303 |
| 13 | ACETOUCH | strong | 0.2255 |
| 14 | ACESWEAR | strong | 0.2253 |
| 15 | EMTSUPRT | strong | 0.2225 |
| 16 | GENHLTH | strong | 0.2202 |
| 17 | ACEADSAF | strong | 0.2202 |
| 18 | CDWORRY | strong | 0.2148 |
| 19 | DIFFALON | strong | 0.2097 |
| 20 | EMPLOY1 | moderate | 0.2082 |
| 21 | CDDISCU1 | strong | 0.1997 |
| 22 | POORHLTH | leakage | 0.1996 |
| 23 | _RFHLTH | leakage | 0.1881 |
| 24 | ACETTHEM | strong | 0.1877 |
| 25 | _LCSYQTS | weak | 0.1876 |
| 26 | ACEPUNCH | strong | 0.1819 |
| 27 | SOFEMALE | weak | 0.1809 |
| 28 | SDHFOOD1 | moderate | 0.1734 |
| 29 | ACEDRINK | strong | 0.1658 |
| 30 | ACEHVSEX | strong | 0.1652 |

## Top-30 by Mutual information

| rank | name | tier | score |
|---|---|---|---|
| 1 | _MENT14D | leakage | 0.0904 |
| 2 | MENTHLTH | leakage | 0.0476 |
| 3 | DECIDE | strong | 0.0464 |
| 4 | POORHLTH | leakage | 0.0367 |
| 5 | _BMI5 | weak | 0.0271 |
| 6 | _PHYS14D | leakage | 0.0241 |
| 7 | GENHLTH | strong | 0.0234 |
| 8 | SSBFRUT3 | weak | 0.0227 |
| 9 | HIVTSTD3 | weak | 0.0199 |
| 10 | EMPLOY1 | moderate | 0.0185 |
| 11 | DIFFALON | strong | 0.0178 |
| 12 | CNCRAGE | weak | 0.0176 |
| 13 | CNCRTYP2 | weak | 0.0162 |
| 14 | LCSNUMC_ | weak | 0.0162 |
| 15 | _RFHLTH | leakage | 0.0160 |
| 16 | MARIJAN1 | moderate | 0.0159 |
| 17 | LSATISFY | strong | 0.0148 |
| 18 | PHYSHLTH | strong | 0.0141 |
| 19 | SDLONELY | strong | 0.0140 |
| 20 | CHILDREN | moderate | 0.0132 |
| 21 | _ASTHMS1 | weak | 0.0119 |
| 22 | ECIGNOW3 | moderate | 0.0118 |
| 23 | _LTASTH1 | weak | 0.0115 |
| 24 | SSBSUGR2 | weak | 0.0114 |
| 25 | ASTHMA3 | strong | 0.0112 |
| 26 | DRNK3GE5 | moderate | 0.0107 |
| 27 | DIABAGE4 | strong | 0.0106 |
| 28 | _CASTHM1 | weak | 0.0100 |
| 29 | HIVTST7 | moderate | 0.0098 |
| 30 | _AIDTST4 | weak | 0.0098 |

## Strong-prior variables not yet in the model

| name | tier | rf | univariate | mi |
|---|---|---|---|---|
| DECIDE | strong | 0.0881 | 0.3405 | 0.0464 |
| DIFFALON | strong | 0.0259 | 0.2097 | 0.0178 |
| PHYSHLTH | strong | 0.0113 | 0.1392 | 0.0141 |
| DIFFWALK | strong | 0.0052 | 0.1423 | 0.0092 |
| CHCCOPD3 | strong | 0.0040 | 0.1245 | 0.0068 |
| DIFFDRES | strong | 0.0039 | 0.1285 | 0.0068 |
| DIABAGE4 | strong | 0.0036 | 0.1302 | 0.0106 |
| EMTSUPRT | strong | 0.0035 | 0.2225 | 0.0073 |
| CIMEMLO1 | strong | 0.0021 | 0.2556 | 0.0037 |
| CDHOUS1 | strong | 0.0017 | 0.2781 | 0.0038 |
| BLIND | strong | 0.0014 | 0.0874 | 0.0034 |
| CDSOCIA1 | strong | 0.0014 | 0.2984 | 0.0036 |
| CVDCRHD4 | strong | 0.0011 | 0.0400 | 0.0008 |
| DEAF | strong | 0.0011 | 0.0345 | 0.0006 |
| CRGVPRB4 | strong | 0.0010 | 0.0826 | 0.0006 |
| CHCOCNC1 | strong | 0.0010 | 0.0173 | 0.0001 |
| CHCKDNY2 | strong | 0.0009 | 0.0524 | 0.0013 |
| CVDSTRK3 | strong | 0.0009 | 0.0500 | 0.0011 |
| CHCSCNC1 | strong | 0.0008 | 0.0196 | 0.0002 |
| ASTHNOW | strong | 0.0008 | 0.0691 | 0.0015 |
| PREDIAB2 | strong | 0.0007 | 0.0652 | 0.0005 |
| ACEADSAF | strong | 0.0007 | 0.2202 | 0.0009 |
| ACEDEPRS | strong | 0.0007 | 0.2827 | 0.0017 |
| ACESWEAR | strong | 0.0006 | 0.2253 | 0.0009 |
| ACETOUCH | strong | 0.0006 | 0.2255 | 0.0011 |
| CAREGIV1 | strong | 0.0006 | 0.0709 | 0.0006 |
| CDDISCU1 | strong | 0.0005 | 0.1997 | 0.0007 |
| ACEHURT1 | strong | 0.0005 | 0.1605 | 0.0004 |
| ACEPUNCH | strong | 0.0005 | 0.1819 | 0.0007 |
| CDWORRY | strong | 0.0004 | 0.2148 | 0.0004 |
| ACEADNED | strong | 0.0004 | 0.1447 | 0.0004 |
| ACETTHEM | strong | 0.0004 | 0.1877 | 0.0008 |
| ACEDRINK | strong | 0.0004 | 0.1658 | 0.0005 |
| ACEDIVRC | strong | 0.0004 | 0.0861 | 0.0001 |
| ACEHVSEX | strong | 0.0003 | 0.1652 | 0.0007 |
| ACEDRUGS | strong | 0.0003 | 0.1650 | 0.0006 |
| CRGVALZD | strong | 0.0003 | 0.0552 | 0.0003 |
| ACEPRISN | strong | 0.0002 | 0.1251 | 0.0003 |

## Top-15 recommended for the next feature set

Combined score: sum of (rf rank + univariate rank + mi rank) across non-skip, non-leakage variables. Lower is better.

| rank | name | tier |
|---|---|---|
| 1 | DECIDE | strong |
| 2 | _BMI5 | weak |
| 3 | GENHLTH | strong |
| 4 | SDLONELY | strong |
| 5 | DIFFALON | strong |
| 6 | EMPLOY1 | moderate |
| 7 | LSATISFY | strong |
| 8 | ECIGNOW3 | moderate |
| 9 | _ASTHMS1 | weak |
| 10 | _LTASTH1 | weak |
| 11 | ASTHMA3 | strong |
| 12 | PHYSHLTH | strong |
| 13 | _CASTHM1 | weak |
| 14 | _SEX | weak |
| 15 | _AIDTST4 | weak |
