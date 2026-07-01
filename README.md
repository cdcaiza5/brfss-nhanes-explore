# BRFSS 2024 — Prediccion de salud mental

Predice una variable binaria de salud mental a partir de 301 variables del
BRFSS 2024. Dos targets disponibles: `ADDEPEV3` (diagnostico de depresion de
por vida) y `_MENT14D` (14+ dias de mala salud mental en los ultimos 30).

## Datos

- Origen: BRFSS 2024, archivo `LLCP2024.XPT_` (codificacion cp1252).
- Tamano: 457 670 registros, 301 columnas.
- Tras limpieza: ~455 000 filas (filas con target NaN se descartan).
- Para los benchmarks: subsample estratificado de 100 000 filas.

## Targets

| Clave      | Variable                | Descripcion                                       | Prevalencia |
|------------|-------------------------|---------------------------------------------------|-------------|
| ADDEPEV3   | diagnostico de depresion | "Alguna vez le dijeron que tuvo depresion"        | 21.1%       |
| _MENT14D   | salud mental reciente   | 14+ dias de mala salud mental en los ultimos 30   | 60.1%       |

`_MENT14D` captura una poblacion distinta: 54 428 personas con 14+ dias
de mala salud mental reciente no tienen diagnostico de por vida (cruce
con `ADDEPEV3`).

## Conjuntos de features

Cuatro presets. Cantidad total de features (sin contar el target).

| Preset           | Total | Numericas | Ordinales | Categoricas | ACE |
|------------------|-------|-----------|-----------|-------------|-----|
| core             | 7     | 1         | 2         | 4           | no  |
| core+ses         | 11    | 1         | 3         | 7           | no  |
| core+ses+aces    | 18    | 2         | 5         | 11          | si  |
| recommended      | 30    | 3         | 6         | 21          | si  |

**`recommended`** es el set auditado. Incluye el set `core+ses+aces` mas:
discapacidad cognitiva/fisica (`DECIDE`, `DIFFWALK`, `DIFFDRES`,
`DIFFALON`, `DEAF`, `BLIND`), salud fisica reciente (`PHYSHLTH`),
condiciones cronicas adicionales (`CHCCOPD3`, `CVDCRHD4`, `CVDSTRK3`),
soporte emocional (`EMTSUPRT`) y raza/etnia (`_RACE`).

Las **ordinales** (`_AGEG5YR`, `EDUCA`, `GENHLTH`, `LSATISFY`, `SDLONELY`,
`EMTSUPRT`) se tratan como numericas (preservan el orden). Las nominales
(`MARITAL`, `EMPLOY1`, `_RACE`) se one-hot-encodian.

## Limpieza

Aplicada en `load_and_clean`:

1. **BMI**: `_BMI5` viene multiplicado por 100 en el XPT. Se divide por 100.
2. **PHYSHLTH y MENTHLTH**: codigo 88 significa "None" (= 0 dias). Se
   mapea a 0 antes de los codigos missing, para no perder la senal
   "0 dias malos".
3. **Codigos missing del codebook** (`MISSING_VALUE_MAP`): 7 = "no sabe",
   9 = "rehuso", 14 = missing en `_AGEG5YR`, 77/88/99 en escalas
   continuas. Se reemplazan por NaN.
4. **Filas con target NaN se descartan** (no se imputa el target).
5. **`ace_score`**: conteo de los 11 items ACE.

### `ace_score` — detalle

Los items ACE tienen dos escalas distintas:

- **Binarios** (5 items: `ACEDEPRS`, `ACEDRINK`, `ACEDRUGS`, `ACEPRISN`,
  `ACEDIVRC`): 1=Si, 2=No. Se cuenta `== 1`.
- **Frecuencia** (6 items: `ACEPUNCH`, `ACEHURT1`, `ACESWEAR`, `ACETOUCH`,
  `ACETTHEM`, `ACEHVSEX`): 1=Nunca, 2=Una vez, 3=Mas de una vez. Se
  cuenta `>= 2`.

Si todos los items ACE son NaN para una fila, `ace_score` queda en NaN
y se imputa con la mediana.

## Modelos

Tres estimadores, mismos hiperparametros en todas las corridas:

| Modelo             | Notas                                                          |
|--------------------|----------------------------------------------------------------|
| RandomForest       | 300 arboles, max_depth=12, n_jobs=-1                           |
| LinearSVC          | Envoltorio CalibratedClassifierCV(cv=3, sigmoid) para probs    |
| LogisticRegression | Solver newton-cholesky, max_iter=2000                          |

No se usa `class_weight="balanced"` (SMOTENC ya balancea la clase
minoritaria en entrenamiento).

## Pipeline por fold

```
datos crudos
  -> load_and_clean (limpia + binariza target)
  -> add_ace_score (si use_ace)
  -> [subsample estratificado, opcional]
  -> StratifiedKFold(cv)
  -> por cada fold:
       imputer (mediana numericas, moda categoricas)
       SMOTENC (sobre-muestrea clase minoritaria)
       encoder (StandardScaler + OneHotEncoder)
       ajustar 3 modelos, predecir, calcular metricas
  -> CSVs por fold + resumen mean +/- std
  -> matriz de confusion acumulada por modelo
```

SMOTENC corre una vez por fold, no por modelo. Esto evita recomputar el
vecino mas cercano 3 veces por fold.

## Metricas

- `accuracy`: fraccion de aciertos.
- `precision` / `recall` / `f1`: sobre la clase positiva.
- `specificity`: `TN / (TN + FP)`.
- `roc_auc`: AUC ROC. Independiente del umbral, robusto al desbalance.
- `pr_auc`: AUC de precision-recall. Mas informativo que ROC-AUC con
  clases desbalanceadas.
- `brier`: Brier score (error cuadratico medio de probabilidad). Mas
  bajo = probabilidades mejor calibradas.

Metrica principal reportada: **ROC-AUC**, complementada con **PR-AUC**
y **Brier**.

## Como correr

```bash
# ADDEPEV3 con set recomendado, 100k filas, 5-fold
python brfss_dataset_model_ADDEPEV3.py --subsample 100000 --cv 5

# _MENT14D con set recomendado, 100k filas, 5-fold
python brfss_dataset_model__MENT14D.py --subsample 100000 --cv 5

# Iteracion rapida con subsample pequeno
python brfss_dataset_model_ADDEPEV3.py --subsample 30000 --cv 2

# Cambiar feature set o target via CLI (solo en el script base)
python brfss_dataset_model.py --target _MENT14D --feature-set core --cv 3

# Full data (sin --subsample) — ~horas por SMOTENC
python brfss_dataset_model_ADDEPEV3.py --cv 5
```

Salidas en `metadata/model_results/`:

- `folds_<target>_<feature_set>.csv` — una fila por (modelo, fold, metrica).
- `summary_<target>_<feature_set>.csv` — mean y std por (modelo, metrica).
- `cm_<modelo>_<target>_<feature_set>.png` — matriz de confusion acumulada.

## Resultados

Benchmark en subsample 100 000, StratifiedKFold(5). Mejor modelo por fila.

### ADDEPEV3 / recommended

| Modelo             | acc   | F1    | ROC-AUC | PR-AUC | Brier |
|--------------------|-------|-------|---------|--------|-------|
| RandomForest       | 0.758 | 0.493 | 0.765   | 0.502  | 0.167 |
| LinearSVC          | 0.723 | 0.479 | 0.743   | 0.485  | 0.184 |
| LogisticRegression | 0.720 | 0.473 | 0.741   | 0.489  | 0.185 |

### _MENT14D / recommended

| Modelo             | acc   | F1    | ROC-AUC | PR-AUC | Brier |
|--------------------|-------|-------|---------|--------|-------|
| RandomForest       | 0.725 | 0.768 | 0.793   | 0.839  | 0.184 |
| LinearSVC          | 0.706 | 0.745 | 0.775   | 0.823  | 0.192 |
| LogisticRegression | 0.709 | 0.752 | 0.775   | 0.823  | 0.191 |

## Estructura del repo

```
brfss_dataset_model.py                # libreria: logica del pipeline + CLI
brfss_dataset_model_ADDEPEV3.py       # wrapper: target=ADDEPEV3, set=recommended
brfss_dataset_model__MENT14D.py       # wrapper: target=_MENT14D, set=recommended
tools/audit.py                        # auditoria de features y comparacion de targets
tools/export_columns.py               # exporta metadata de columnas a CSV/MD
requirements.txt                      # dependencias Python
metadata/
  columns/
    brfss_2024_columns.csv            # 301 filas: position, name, label
    brfss_2024_columns.md
  audit/
    feature_audit.csv                 # 300 filas: tier + rf + univariate + mi
    feature_audit.md
    target_comparison.csv             # 4 targets: prevalence, AUC, F1
    target_comparison.md
  model_results/
    folds_<target>_<set>.csv
    summary_<target>_<set>.csv
    cm_<modelo>_<target>_<set>.png
```

## Auditoria de features

`metadata/audit/feature_audit.md` y `metadata/audit/target_comparison.md` contienen
el detalle de la seleccion de features y la eleccion del target. El
proceso:

1. Clasificacion manual de las 301 variables en tiers (strong / moderate
   / weak / leakage / skip) segun literatura de BRFSS.
2. Tres rankings empiricos: RandomForest importance, asociacion
   univariada (point-biserial / V de Cramer), informacion mutua.
3. Combinacion de los rankings para producir el set `recommended`.

Para el target, se evaluaron 4 candidatos con un baseline LR de 3-fold.
`_MENT14D` resulto mejor en PR-AUC y F1, y captura una poblacion
distinta a `ADDEPEV3` (ver cruzamiento en el report).

## Dependencias

Ver `requirements.txt`:

```
pyreadstat
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
```

Instalacion:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Limitaciones

- Subsample de 100 000 para benchmarks. Full data: ~horas por SMOTENC.
- Sin tuning de hiperparametros (default de sklearn).
- Sin XGBoost / LightGBM (no instalados).
- `_RACE` se one-hot-encodea; target encoding seria ligeramente mejor.
- El "always no" baseline para ADDEPEV3 es 79% accuracy; los modelos
  quedan por debajo de ese numero porque SMOTENC los empuja a predecir
  positivo. Para screening, el recall (0.55-0.65 en ADDEPEV3) es la
  metrica relevante.
