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
(`MARITAL`, `EMPLOY1`) se one-hot-encodian. `_RACE` se trata aparte: ver
la seccion siguiente.

### Raza/etnia (`--include-race`)

`_RACE` no esta en el set `recommended` por default. Es un flag opt-in
porque es la unica variable con scope US-specifico y su inclusion debe
ser explicita.

- **Sin el flag**: `_RACE` no aparece en el dataset, ni en X ni en el
  modelo. Los conteos de features son los de la tabla de arriba.
- **Con el flag**: `_RACE` se agrega como mean-encoding por fold
  (ajustado solo sobre el train de cada fold, sin fuga). Reemplaza 7
  dummies OHE por 1 columna continua.

Lift medido (ver `reports/benchmark_2x2x2.md`):

| Target | LightGBM ROC-AUC | Δ |
|---|---:|---:|
| `_MENT14D` | 0.7958 → 0.7986 | **+0.0028** |
| `ADDEPEV3` | 0.7780 → 0.7890 | **+0.0110** |

El lift es uniforme entre feature sets: `_RACE` aporta senal ortogonal
a las features existentes, no redundante. En modelos lineales el efecto
es despreciable (mean-encoding no es monotono).

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

Cinco estimadores, mismos hiperparametros en todas las corridas:

| Modelo             | Notas                                                          |
|--------------------|----------------------------------------------------------------|
| RandomForest       | 300 arboles, max_depth=12, n_jobs=-1                           |
| LinearSVC          | Envoltorio CalibratedClassifierCV(cv=3, sigmoid) para probs    |
| LogisticRegression | Solver newton-cholesky, max_iter=2000                          |
| XGBoost            | 300 arboles, max_depth=6, learning_rate=0.1, n_jobs=-1         |
| LightGBM           | 300 arboles, learning_rate=0.1, n_jobs=-1                      |

Se usa `class_weight="balanced"` por defecto (recomendado por el benchmark
`reports/balancing_comparison.md`): repondera losses sin resamplear, es
5-7x mas rapido que SMOTENC y gana en ROC-AUC para todos los modelos
sobre la mejor combo del 2x2x2. Otras opciones via `--balancing`: smote,
undersampling-random, undersampling-nearmiss.

## Pipeline por fold

```
datos crudos
  -> load_and_clean (limpia + binariza target)
  -> add_ace_score (si use_ace)
  -> [subsample estratificado, opcional]
  -> StratifiedKFold(cv)
  -> por cada fold:
       imputer (mediana numericas, moda categoricas)
       resampleo o reponderado segun --balancing (ver tabla abajo)
       encoder (StandardScaler + OneHotEncoder)
       ajustar 5 modelos, predecir, calcular metricas
  -> CSVs por fold + resumen mean +/- std
  -> matriz de confusion acumulada por modelo
```

### Balanceo de clases (`--balancing`)

| Valor | Mecanismo |
|---|---|
| `class_weight` (default) | No resamplea. Pasa `class_weight="balanced"` a cada modelo (incluido XGBoost, que lo mapea a `scale_pos_weight` internamente). |
| `smote` | SMOTENC sobre-muestrea la clase minoritaria con sintesis KNN. |
| `undersampling-random` | `RandomUnderSampler` submuestrea la clase mayoritaria. |
| `undersampling-nearmiss` | `NearMiss(version=1)` submuestrea por distancia a la minoritaria. |

El resampleo corre una vez por fold, no por modelo. Esto evita recomputar
el vecino mas cercano 5 veces por fold.

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

# Full data (sin --subsample) — ~30 min con --balancing class_weight
python brfss_dataset_model_ADDEPEV3.py --cv 5

# Probar otra estrategia de balanceo (smote, undersampling-random, nearmiss)
python brfss_dataset_model__MENT14D.py --balancing smote --subsample 100000 --cv 5
python brfss_dataset_model_ADDEPEV3.py --balancing undersampling-nearmiss --subsample 100000 --cv 5
```

### Recomendaciones (de los benchmarks en `reports/`)

- **Mejor combinacion por defecto**: `_MENT14D` + `recommended` +
  `--include-race` + `--balancing class_weight`. Reproducir con:
  ```bash
  python brfss_dataset_model__MENT14D.py \
    --include-race --subsample 100000 --cv 5
  ```
  Espera ~2 min en CPU moderna. LightGBM lidera con ROC-AUC ~0.80 y
  F1 ~0.79.

- **Para `ADDEPEV3`** (target mas dificil por base rate 21% y senal
  historica): recall es ~0.40 con todos los modelos; ROC-AUC ~0.79 con
  LightGBM + race + class_weight. Ver `reports/benchmark_2x2x2.md`.

- **Smoke test rapido** (~30 s): agregar `--subsample 30000 --cv 2`.

- **Sin tuning de hiperparametros**: los defaults de sklearn/XGBoost
  funcionan bien; el lift medido viene del feature set y la estrategia
  de balanceo, no del tuning.

Salidas en `metadata/model_results/`:

- `folds_<target>_<feature_set>_<balancing>.csv` — una fila por (modelo, fold, metrica).
- `summary_<target>_<feature_set>_<balancing>.csv` — mean y std por (modelo, metrica).
- `cm_<modelo>_<target>_<feature_set>_<balancing>.png` — matriz de confusion acumulada.

## Resultados

Los benchmarks detallados viven en `reports/`:

- **`reports/benchmark_2x2x2.md`** — 8 corridas (2 targets × 2 feature sets
  × 2 estados de `--include-race`), 100k subsample, 5-fold CV.
  Headline: `_MENT14D` + `recommended` + `--include-race` con LightGBM
  alcanza ROC-AUC 0.7986 y F1 0.7904 — la mejor combo del barrido.

- **`reports/balancing_comparison.md`** — 4 estrategias de balanceo
  sobre la mejor combo del 2x2x2. Headline: `class_weight` gana en
  ROC-AUC para todos los modelos y es 5-7x mas rapido que `smote`.

Metrica principal: **ROC-AUC**, complementada con **PR-AUC** y **Brier**.

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
    folds_<target>_<set>_<balancing>.csv
    summary_<target>_<set>_<balancing>.csv
    cm_<modelo>_<target>_<set>_<balancing>.png
reports/
  benchmark_2x2x2.md                  # 8 corridas: 2 targets x 2 sets x 2 race
  balancing_comparison.md             # 4 estrategias de balanceo sobre la mejor combo
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
xgboost
lightgbm
```

Instalacion:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Limitaciones

- Subsample de 100 000 para benchmarks. Full data con `--balancing
  class_weight` (default) corre en ~30 min; con `smote` o
  `undersampling-nearmiss` seria ~5-7x mas lento por el KNN del
  resampleo.
- Sin tuning de hiperparametros (defaults de sklearn / XGBoost /
  LightGBM). El lift medido en los benchmarks viene del feature set y la
  estrategia de balanceo, no del tuning.
- `ADDEPEV3` (depresion de por vida) es un target dificil: base rate
  21%, recall ~0.40 con todos los modelos, AUC ~0.79. El "always no"
  baseline da 79% accuracy, pero eso enmascara el problema (recall ~0).
  Para screening, el recall es la metrica relevante y sigue siendo bajo.
- `_MENT14D` (14+ dias de mala salud mental reciente) es un target mas
  facil: base rate 60%, senal current-state, AUC ~0.80, F1 ~0.79.
- `_RACE` es opt-in via `--include-race`. Cuando se incluye, es
  mean-encoded (no one-hot) para preservar el sample size.
- Threshold fijo en 0.5. Para `ADDEPEV3` ajustarlo puede mejorar recall
  a costa de precision, pero el limite real es la senal insuficiente.
