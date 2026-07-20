# Seleccion del preset `default` — 2026-07-20

Seleccion de las 30 mejores variables para `ADDEPEV3` sobre el universo
completo de columnas (menos ID/geo/peso/estructurales y el target).
Generado por `tools/select_rf_top30.py` (seed 42, subsample 100,000).

## Techo de ranking

- RF sobre el universo completo (train): **0.9039**
- RF sobre el universo completo (test): **0.8407**
- RF sobre el top-30 canonico (test): **0.8331**
  (delta vs universo: -0.0077)

## TOP 30 — RandomForest (canonico)

| rank | variable | importancia RF |
|---|---|---|
| 1 | `MENTHLTH` | 0.15857 |
| 2 | `_MENT14D` | 0.14782 |
| 3 | `POORHLTH` | 0.07117 |
| 4 | `DECIDE` | 0.05926 |
| 5 | `_PHYS14D` | 0.02364 |
| 6 | `PHYSHLTH` | 0.02300 |
| 7 | `GENHLTH` | 0.01447 |
| 8 | `SDLONELY` | 0.01446 |
| 9 | `DIFFALON` | 0.01355 |
| 10 | `_RFHLTH` | 0.01305 |
| 11 | `_SEX` | 0.01217 |
| 12 | `SEXVAR` | 0.01164 |
| 13 | `ECIGNOW3` | 0.01133 |
| 14 | `_AGE80` | 0.01007 |
| 15 | `EMPLOY1` | 0.00950 |
| 16 | `ASTHMA3` | 0.00900 |
| 17 | `_ASTHMS1` | 0.00887 |
| 18 | `_AGEG5YR` | 0.00877 |
| 19 | `_BMI5` | 0.00876 |
| 20 | `LSATISFY` | 0.00815 |
| 21 | `HIVTST7` | 0.00791 |
| 22 | `_LTASTH1` | 0.00717 |
| 23 | `_DRDXAR2` | 0.00714 |
| 24 | `CELLSEX3` | 0.00712 |
| 25 | `_AIDTST4` | 0.00676 |
| 26 | `HIVTSTD3` | 0.00667 |
| 27 | `HEIGHT3` | 0.00649 |
| 28 | `HAVARTH4` | 0.00638 |
| 29 | `WTKG3` | 0.00637 |
| 30 | `_STSTR` | 0.00632 |

## TOP 30 — Mutual Information (contraste)

| rank | variable | MI |
|---|---|---|
| 1 | `_BMI5` | 0.14901 |
| 2 | `_STSTR` | 0.14204 |
| 3 | `MENTHLTH` | 0.10023 |
| 4 | `_MENT14D` | 0.09472 |
| 5 | `POORHLTH` | 0.04895 |
| 6 | `DECIDE` | 0.04857 |
| 7 | `HIVTSTD3` | 0.04708 |
| 8 | `WTKG3` | 0.03341 |
| 9 | `PHYSHLTH` | 0.02618 |
| 10 | `GENHLTH` | 0.02242 |
| 11 | `_PHYS14D` | 0.02221 |
| 12 | `EMPLOY1` | 0.01908 |
| 13 | `_AGE80` | 0.01819 |
| 14 | `DIFFALON` | 0.01818 |
| 15 | `_PACKYRS` | 0.01610 |
| 16 | `_RFHLTH` | 0.01521 |
| 17 | `_ASTHMS1` | 0.01358 |
| 18 | `MEDCOST1` | 0.01357 |
| 19 | `_LTASTH1` | 0.01345 |
| 20 | `ASTHMA3` | 0.01345 |
| 21 | `HIVTST7` | 0.01322 |
| 22 | `_AIDTST4` | 0.01322 |
| 23 | `DIABAGE4` | 0.01293 |
| 24 | `_AGEG5YR` | 0.01232 |
| 25 | `_LCSYQTS` | 0.01219 |
| 26 | `_SEX` | 0.01195 |
| 27 | `SEXVAR` | 0.01192 |
| 28 | `ECIGNOW3` | 0.01189 |
| 29 | `HEIGHT3` | 0.01189 |
| 30 | `_DUALCOR` | 0.01163 |

## Permutation importance (RF, ROC-AUC drop, top-30 RF)

| variable | ROC-AUC drop |
|---|---|
| `MENTHLTH` | 0.03593 |
| `_MENT14D` | 0.02084 |
| `POORHLTH` | 0.00444 |
| `DECIDE` | 0.01438 |
| `_PHYS14D` | -0.00003 |
| `PHYSHLTH` | 0.00014 |
| `GENHLTH` | 0.00108 |
| `SDLONELY` | 0.00223 |
| `DIFFALON` | 0.00071 |
| `_RFHLTH` | 0.00007 |
| `_SEX` | 0.00154 |
| `SEXVAR` | 0.00189 |
| `ECIGNOW3` | 0.00211 |
| `_AGE80` | 0.00169 |
| `EMPLOY1` | 0.00085 |
| `ASTHMA3` | 0.00049 |
| `_ASTHMS1` | 0.00055 |
| `_AGEG5YR` | 0.00102 |
| `_BMI5` | 0.00066 |
| `LSATISFY` | 0.00068 |
| `HIVTST7` | 0.00081 |
| `_LTASTH1` | 0.00040 |
| `_DRDXAR2` | 0.00163 |
| `CELLSEX3` | 0.00078 |
| `_AIDTST4` | 0.00071 |
| `HIVTSTD3` | 0.00022 |
| `HEIGHT3` | 0.00076 |
| `HAVARTH4` | 0.00167 |
| `WTKG3` | 0.00054 |
| `_STSTR` | 0.00049 |

## Overlap RF ∩ MI

25 / 30 compartidas entre el top-30 de RF y el top-30 de MI.

## Set canonico (top-30 RF)

```
`MENTHLTH`, `_MENT14D`, `POORHLTH`, `DECIDE`, `_PHYS14D`, `PHYSHLTH`, `GENHLTH`, `SDLONELY`, `DIFFALON`, `_RFHLTH`, `_SEX`, `SEXVAR`, `ECIGNOW3`, `_AGE80`, `EMPLOY1`, `ASTHMA3`, `_ASTHMS1`, `_AGEG5YR`, `_BMI5`, `LSATISFY`, `HIVTST7`, `_LTASTH1`, `_DRDXAR2`, `CELLSEX3`, `_AIDTST4`, `HIVTSTD3`, `HEIGHT3`, `HAVARTH4`, `WTKG3`, `_STSTR`
```

## Notas

- El canonico es el top-30 de RF. MI sirve de contraste; tiende a
  contaminarse con pesos de diseno y variables estructurales, por eso
  no se usa como set final.
- `_RACE` se mean-encodea aparte en el pipeline (no entra en las 30).
- El preset `default` en `brfss_dataset_model.py` aplica dos sustituciones
  manuales sobre este top-30: quita `_MENT14D` (casi-duplicado del
  target) y `_STSTR` (estrato de muestreo), y agrega `_PACKYRS` y
  `MEDCOST1` del rango RF siguiente.
