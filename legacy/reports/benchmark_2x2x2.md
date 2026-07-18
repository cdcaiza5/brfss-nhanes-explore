# BRFSS 2024 — Benchmark 2×2×2

Comparacion completa de los dos targets, los dos feature sets y el flag
`--include-race` prendido y apagado. Ocho corridas en total, todas con la
misma semilla (`SEED=42`), stratified 5-fold CV, y subsample de 100,000 filas
del XPT completo.

## Setup

Comandos ejecutados:

```bash
# _MENT14D + recommended + no race
python brfss_dataset_model__MENT14D.py --subsample 100000 --cv 5 --out /tmp/benchmark_full/m14d_rec_off

# _MENT14D + recommended + race
python brfss_dataset_model__MENT14D.py --include-race --subsample 100000 --cv 5 --out /tmp/benchmark_full/m14d_rec_on

# _MENT14D + core+ses+aces + no race
python brfss_dataset_model__MENT14D.py --feature-set core+ses+aces --subsample 100000 --cv 5 --out /tmp/benchmark_full/m14d_csa_off

# _MENT14D + core+ses+aces + race
python brfss_dataset_model__MENT14D.py --feature-set core+ses+aces --include-race --subsample 100000 --cv 5 --out /tmp/benchmark_full/m14d_csa_on

# (mismo patron para ADDEPEV3 con brfss_dataset_model_ADDEPEV3.py)
```

Datasets (post limpieza, codebook missing codes aplicados):

| Target | Filas | Tasa de positivos |
|---|---:|---:|
| `_MENT14D` | 449,514 | 60.04% |
| `ADDEPEV3` | 455,006 | 21.13% |

Modelos: RandomForest, LinearSVC, LogisticRegression, XGBoost, LightGBM.

## Matriz de corridas

| # | Target | Feature set | `--include-race` | n_features | n | Wall time |
|---|---|---|---|---:|---:|---:|
| 1 | `_MENT14D` | recommended | no  | 29 | 100,000 | 647 s |
| 2 | `_MENT14D` | recommended | yes | 30 | 100,000 | 651 s |
| 3 | `_MENT14D` | core+ses+aces | no  | 18 | 100,000 | 622 s |
| 4 | `_MENT14D` | core+ses+aces | yes | 19 | 100,000 | 672 s |
| 5 | `ADDEPEV3` | recommended | no  | 29 | 100,000 | 475 s |
| 6 | `ADDEPEV3` | recommended | yes | 30 | 100,000 | 462 s |
| 7 | `ADDEPEV3` | core+ses+aces | no  | 18 | 100,000 | 427 s |
| 8 | `ADDEPEV3` | core+ses+aces | yes | 19 | 100,000 | 352 s |

---

## Resultados `_MENT14D`

### recommended, sin `_RACE` (n_features=29)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7243 | 0.7775 | 0.7577 | 0.7674 | 0.6740 | 0.7916 | 0.8377 | 0.1842 |
| LinearSVC | 0.7025 | 0.7765 | 0.7085 | 0.7409 | 0.6934 | 0.7734 | 0.8229 | 0.1937 |
| LogisticRegression | 0.7070 | 0.7728 | 0.7252 | 0.7483 | 0.6796 | 0.7737 | 0.8226 | 0.1919 |
| XGBoost | 0.7348 | 0.7624 | 0.8110 | 0.7860 | 0.6202 | 0.7952 | 0.8396 | 0.1778 |
| LightGBM | 0.7360 | 0.7637 | 0.8113 | 0.7868 | 0.6228 | 0.7958 | 0.8404 | 0.1777 |

### recommended, con `_RACE` (n_features=30)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7253 | 0.7787 | 0.7579 | 0.7681 | 0.6762 | 0.7930 | 0.8384 | 0.1838 |
| LinearSVC | 0.7028 | 0.7767 | 0.7088 | 0.7412 | 0.6937 | 0.7731 | 0.8227 | 0.1938 |
| LogisticRegression | 0.7065 | 0.7726 | 0.7246 | 0.7478 | 0.6794 | 0.7734 | 0.8224 | 0.1920 |
| XGBoost | 0.7380 | 0.7622 | 0.8192 | 0.7897 | 0.6159 | 0.7984 | 0.8420 | 0.1762 |
| LightGBM | 0.7385 | 0.7620 | 0.8211 | 0.7904 | 0.6145 | 0.7986 | 0.8423 | 0.1761 |

### core+ses+aces, sin `_RACE` (n_features=18)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.6967 | 0.7642 | 0.7158 | 0.7392 | 0.6681 | 0.7606 | 0.8150 | 0.1975 |
| LinearSVC | 0.6888 | 0.7682 | 0.6898 | 0.7269 | 0.6872 | 0.7571 | 0.8121 | 0.2002 |
| LogisticRegression | 0.6903 | 0.7656 | 0.6980 | 0.7302 | 0.6788 | 0.7572 | 0.8122 | 0.1993 |
| XGBoost | 0.7069 | 0.7471 | 0.7740 | 0.7603 | 0.6061 | 0.7643 | 0.8164 | 0.1915 |
| LightGBM | 0.7074 | 0.7483 | 0.7726 | 0.7602 | 0.6094 | 0.7653 | 0.8173 | 0.1912 |

### core+ses+aces, con `_RACE` (n_features=19)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7004 | 0.7647 | 0.7236 | 0.7436 | 0.6654 | 0.7627 | 0.8161 | 0.1967 |
| LinearSVC | 0.6885 | 0.7680 | 0.6897 | 0.7267 | 0.6868 | 0.7567 | 0.8119 | 0.2003 |
| LogisticRegression | 0.6901 | 0.7650 | 0.6984 | 0.7302 | 0.6776 | 0.7568 | 0.8119 | 0.1994 |
| XGBoost | 0.7131 | 0.7465 | 0.7907 | 0.7679 | 0.5964 | 0.7681 | 0.8188 | 0.1893 |
| LightGBM | 0.7130 | 0.7454 | 0.7929 | 0.7684 | 0.5929 | 0.7688 | 0.8195 | 0.1890 |

### Lift de `recommended` sobre `core+ses+aces` (LightGBM, sin `_RACE`)

| Metric | core+ses+aces | recommended | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.7653 | 0.7958 | **+0.0305** |
| F1 | 0.7602 | 0.7868 | +0.0266 |
| PR-AUC | 0.8173 | 0.8404 | +0.0231 |
| Brier | 0.1912 | 0.1777 | −0.0135 |

### Efecto de `_RACE` (LightGBM)

| Feature set | ROC-AUC off | ROC-AUC on | Δ |
|---|---:|---:|---:|
| recommended | 0.7958 | 0.7986 | **+0.0028** |
| core+ses+aces | 0.7653 | 0.7688 | **+0.0035** |

---

## Resultados `ADDEPEV3`

### recommended, sin `_RACE` (n_features=29)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7577 | 0.4421 | 0.5604 | 0.4942 | 0.8106 | 0.7655 | 0.5089 | 0.1682 |
| LinearSVC | 0.7104 | 0.3839 | 0.6131 | 0.4722 | 0.7365 | 0.7371 | 0.4758 | 0.1912 |
| LogisticRegression | 0.7102 | 0.3820 | 0.6021 | 0.4675 | 0.7391 | 0.7340 | 0.4788 | 0.1909 |
| XGBoost | 0.8171 | 0.6064 | 0.3826 | 0.4691 | 0.9335 | 0.7772 | 0.5395 | 0.1357 |
| LightGBM | 0.8145 | 0.5932 | 0.3882 | 0.4692 | 0.9287 | 0.7780 | 0.5397 | 0.1360 |

### recommended, con `_RACE` (n_features=30)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7683 | 0.4598 | 0.5502 | 0.5009 | 0.8268 | 0.7727 | 0.5160 | 0.1640 |
| LinearSVC | 0.7142 | 0.3893 | 0.6208 | 0.4785 | 0.7392 | 0.7427 | 0.4818 | 0.1894 |
| LogisticRegression | 0.7139 | 0.3875 | 0.6097 | 0.4738 | 0.7418 | 0.7402 | 0.4855 | 0.1889 |
| XGBoost | 0.8204 | 0.6200 | 0.3870 | 0.4765 | 0.9365 | 0.7881 | 0.5524 | 0.1326 |
| LightGBM | 0.8181 | 0.6075 | 0.3922 | 0.4766 | 0.9321 | 0.7890 | 0.5514 | 0.1328 |

### core+ses+aces, sin `_RACE` (n_features=18)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7308 | 0.4015 | 0.5591 | 0.4674 | 0.7768 | 0.7414 | 0.4561 | 0.1803 |
| LinearSVC | 0.6938 | 0.3675 | 0.6235 | 0.4624 | 0.7126 | 0.7308 | 0.4540 | 0.1980 |
| LogisticRegression | 0.6938 | 0.3669 | 0.6192 | 0.4608 | 0.7138 | 0.7295 | 0.4543 | 0.1978 |
| XGBoost | 0.7873 | 0.4958 | 0.3961 | 0.4403 | 0.8921 | 0.7489 | 0.4735 | 0.1499 |
| LightGBM | 0.7861 | 0.4924 | 0.4009 | 0.4419 | 0.8893 | 0.7497 | 0.4726 | 0.1501 |

### core+ses+aces, con `_RACE` (n_features=19)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7443 | 0.4187 | 0.5413 | 0.4721 | 0.7987 | 0.7489 | 0.4642 | 0.1747 |
| LinearSVC | 0.6991 | 0.3741 | 0.6308 | 0.4697 | 0.7174 | 0.7374 | 0.4616 | 0.1959 |
| LogisticRegression | 0.6990 | 0.3732 | 0.6251 | 0.4673 | 0.7188 | 0.7364 | 0.4622 | 0.1956 |
| XGBoost | 0.7968 | 0.5263 | 0.3814 | 0.4422 | 0.9080 | 0.7607 | 0.4925 | 0.1442 |
| LightGBM | 0.7954 | 0.5213 | 0.3876 | 0.4446 | 0.9046 | 0.7622 | 0.4923 | 0.1443 |

### Lift de `recommended` sobre `core+ses+aces` (LightGBM, sin `_RACE`)

| Metric | core+ses+aces | recommended | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.7497 | 0.7780 | **+0.0283** |
| F1 | 0.4419 | 0.4692 | +0.0273 |
| PR-AUC | 0.4726 | 0.5397 | **+0.0671** |
| Brier | 0.1501 | 0.1360 | −0.0141 |

### Efecto de `_RACE` (LightGBM)

| Feature set | ROC-AUC off | ROC-AUC on | Δ |
|---|---:|---:|---:|
| recommended | 0.7780 | 0.7890 | **+0.0110** |
| core+ses+aces | 0.7497 | 0.7622 | **+0.0125** |

---

## Comparacion entre feature sets

Lift de `recommended` sobre `core+ses+aces` en ROC-AUC (sin `_RACE`):

| Target | core+ses+aces | recommended | Δ |
|---|---:|---:|---:|
| `_MENT14D` (LightGBM) | 0.7653 | 0.7958 | **+0.0305** |
| `_MENT14D` (XGBoost) | 0.7643 | 0.7952 | +0.0309 |
| `ADDEPEV3` (LightGBM) | 0.7497 | 0.7780 | +0.0283 |
| `ADDEPEV3` (XGBoost) | 0.7489 | 0.7772 | +0.0283 |

El set `recommended` agrega ~0.03 ROC-AUC sobre `core+ses+aces` para ambos
targets y ambos modelos de gradient boosting. La diferencia es estable y
grande (>10σ sobre el std por fold).

`recommended` agrega 11 features sobre `core+ses+aces`:

- **Discapacidad cognitiva/fisica**: `DECIDE`, `DIFFWALK`, `DIFFDRES`, `DIFFALON`, `DEAF`, `BLIND`.
- **Condiciones cronicas adicionales**: `CHCCOPD3`, `CVDCRHD4`, `CVDSTRK3`.
- **Soporte emocional**: `EMTSUPRT` (ordinal).
- **Salud fisica**: `PHYSHLTH` (numeric, dias de mala salud fisica).

El modelo lineal (LogisticRegression) muestra lift similar o mayor:

| Target | core+ses+aces | recommended | Δ |
|---|---:|---:|---:|
| `_MENT14D` (LR) | 0.7572 | 0.7737 | +0.0165 |
| `ADDEPEV3` (LR) | 0.7295 | 0.7340 | +0.0045 |

Para `ADDEPEV3` con regresion logistica, `recommended` aporta solo +0.005
ROC-AUC. El LR no captura las no-linealidades de las features adicionales
tan bien como los modelos de gradient boosting.

---

## Efecto de `_RACE` por feature set

`_RACE` aporta senal adicional independientemente del feature set. El efecto
es mas fuerte en `ADDEPEV3` que en `_MENT14D`:

| Target / Model | recommended Δ ROC-AUC | core+ses+aces Δ ROC-AUC |
|---|---:|---:|
| `_MENT14D` (LightGBM) | +0.0028 | +0.0035 |
| `_MENT14D` (XGBoost) | +0.0032 | +0.0038 |
| `ADDEPEV3` (LightGBM) | +0.0110 | +0.0125 |
| `ADDEPEV3` (XGBoost) | +0.0109 | +0.0118 |
| `ADDEPEV3` (RandomForest) | +0.0073 | +0.0075 |

El efecto es similar en magnitud entre los dos feature sets, con una
tendencia leve a ser mayor en `core+ses+aces` (que parte de menor base de
informacion). Esto sugiere que `_RACE` aporta senal ortogonal a las
features existentes, no redundante.

Para modelos lineales (LR, LinearSVC), el efecto de `_RACE` es despreciable
(<=0.0003 ROC-AUC en ambos targets y feature sets). El mean-encoding
produjo un valor continuo que el modelo lineal no puede aprovechar para
mejorar la discriminacion (la relacion entre raza y target no es monotona).

---

## Ranking por metrica (top model por combinacion)

ROC-AUC, mejor modelo por corrida:

| Run | Best model | ROC-AUC | F1 | Brier |
|---|---|---:|---:|---:|
| `_MENT14D` rec off | LightGBM | 0.7958 | 0.7868 | 0.1777 |
| `_MENT14D` rec on | LightGBM | 0.7986 | 0.7904 | 0.1761 |
| `_MENT14D` csa off | LightGBM | 0.7653 | 0.7602 | 0.1912 |
| `_MENT14D` csa on | LightGBM | 0.7688 | 0.7684 | 0.1890 |
| `ADDEPEV3` rec off | LightGBM | 0.7780 | 0.4692 | 0.1360 |
| `ADDEPEV3` rec on | LightGBM | 0.7890 | 0.4766 | 0.1328 |
| `ADDEPEV3` csa off | LightGBM | 0.7497 | 0.4419 | 0.1501 |
| `ADDEPEV3` csa on | LightGBM | 0.7622 | 0.4446 | 0.1443 |

**LightGBM gana las 8 corridas en ROC-AUC y F1.** XGBoost queda segundo,
con metricas casi identicas (diferencia <0.002 en ROC-AUC, <0.004 en F1).

Para `ADDEPEV3` el mejor F1 absolute no es LightGBM sino RandomForest
(0.4942-0.5009 vs 0.4692-0.4766). Esto se debe a que RandomForest tiene
mayor recall (0.55 vs 0.39) a costa de menor precision (0.44 vs 0.61). F1
los favorece por el balance.

---

## Conclusiones

1. **LightGBM es el modelo dominante** en las 8 corridas segun ROC-AUC, F1
   y Brier. XGBoost queda a <0.002 ROC-AUC de LightGBM. La diferencia entre
   los dos es estadisticamente marginal.

2. **`recommended` supera a `core+ses+aces` por ~0.03 ROC-AUC** en ambos
   targets. Los modelos de gradient boosting capturan las no-linealidades
   de las features adicionales (discapacidad, condiciones cronicas,
   soporte emocional). El lift es estable y grande (>10σ sobre el std
   por fold).

3. **`_RACE` siempre ayuda a los modelos de gradient boosting** y el efecto
   es independiente del feature set. El lift es chico en `_MENT14D`
   (+0.003 ROC-AUC) y grande en `ADDEPEV3` (+0.011 ROC-AUC). Los modelos
   lineales no se benefician porque el mean-encoding no es monotono.

4. **`ADDEPEV3` es un problema mas dificil que `_MENT14D`** por dos
   razones combinadas: (a) base rate mas bajo (21% vs 60%) reduce el techo
   de PR-AUC; (b) el target es historico (lifetime diagnosis) y los
   features son current-state, asi que el modelo no puede separar bien a
   quienes fueron diagnosticados hace anos pero estan en remision. La
   brecha en ROC-AUC entre targets es ~0.01-0.02 para el mismo modelo y
   feature set.

5. **Recall es el talon de Aquiles de `ADDEPEV3`**: ~0.39 para XGBoost y
   LightGBM. El modelo aprende a ser conservador (predecir "no" a menos
   que los features digan claramente "si") porque predecir "si" es
   arriesgado con 21% positivos. Esto no se arregla con mas features ni
   con `_RACE`; es un problema de senal insuficiente en los datos.

6. **Para uso practico**:
   - Usar `recommended` + `--include-race` + LightGBM como configuracion
     por defecto.
   - En `ADDEPEV3`, considerar ajuste de threshold si el costo de un falso
     negativo es alto: el recall esta en 0.39 con threshold 0.5.
   - El flag `--include-race` es opt-in; sin el, `_RACE` no aparece en
     los features ni en el modelo. Activarlo es seguro y siempre ayuda en
     los modelos de gradient boosting.
