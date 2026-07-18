# BRFSS 2024 — Prediccion de depresion

Predice el diagnostico de depresion de por vida (`ADDEPEV3`) a partir de 301
variables del BRFSS 2024.

## Datos

- Origen: BRFSS 2024, archivo `LLCP2024.XPT_` (codificacion cp1252).
- Tamano: 457 670 registros, 301 columnas.
- Tras limpieza: ~455 000 filas (filas con target NaN se descartan).
- Para los benchmarks: subsample estratificado de 100 000 filas.

## Target

| Clave      | Variable                | Descripcion                                       | Prevalencia |
|------------|-------------------------|---------------------------------------------------|-------------|
| ADDEPEV3   | diagnostico de depresion | "Alguna vez le dijeron que tuvo depresion"        | 21.1%       |

`ADDEPEV3` es la unica variable de diagnostico de depresion en las 301
columnas del BRFSS 2024. Es un target dificil: base rate 21% y senal
historica, con recall ~0.40 y ROC-AUC ~0.79.

## Conjuntos de features

Un unico preset: **`default`**. 30 predictores seleccionados sobre el
universo completo de columnas del BRFSS 2024 (301 menos ID/geo/peso y el
target) con RandomForest (`tools/select_rf_top30.py`, 100k subsample, RF
balanceado); mutual information y permutation importance se usaron como
contraste. Se excluyeron `_MENT14D` (casi-duplicado del target ADDEPEV3) y
`_STSTR` (estrato de muestreo, no predictor real). `_RACE` se agrega
aparta como columna mean-encoded (no cuenta dentro de las 30).

| Preset  | Total predictores | Numericas | Ordinales | Categoricas | `_RACE` |
|---------|-------------------|-----------|-----------|-------------|---------|
| default | 30                | 8         | 4         | 18          | si (mean-encoded) |

El set canonico da ROC-AUC ~0.833 en hold-out (RF, 100k), casi igual al
universo completo de 293 columnas (~0.840). Las variables de sintoma de
salud (`MENTHLTH`, `POORHLTH`, `_PHYS14D`, `_RFHLTH`) estan incluidas: no
son leakage del target (diagnostico de por vida) pero inflan el
performance por ser del mismo constructo.

Las **numericas** (`_BMI5`, `_AGE80`, `PHYSHLTH`, `MENTHLTH`, `POORHLTH`,
`HEIGHT3`, `WTKG3`, `_PACKYRS`) y **ordinales** (`_AGEG5YR`, `GENHLTH`,
`SDLONELY`, `LSATISFY`) se tratan como continuas (mediana + StandardScaler).
Las **categoricas** se one-hot-encodian (`drop="if_binary"`). `_RACE` se
trata como mean-encoded (1 columna continua, ajustada por fold).

### Raza/etnia (`_RACE`)

`_RACE` se incluye por defecto como mean-encoding por fold (ajustado solo
sobre el train de cada fold, sin fuga de informacion). Reemplaza 7 dummies
one-hot por 1 columna continua. El lift es ortogonal a las otras features;
en modelos lineales el efecto es despreciable (mean-encoding no es
monotono).

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

Se probo un Ensemble (StackingClassifier sobre las 5 bases, meta-learner
LogReg) y se descarto: en full data empata a XGBoost/LightGBM (ROC-AUC
0.843) sin superarlos y no afecta las metricas peores; cuesta ~3-4x el
fit por fold. Ver `reports/changes_and_results.md`.

Se usa `class_weight="balanced"` por defecto: repondera losses sin
resamplear, es 5-7x mas rapido que SMOTENC y gana en ROC-AUC para todos
los modelos. Otras opciones via `--balancing`: smote,
undersampling-random, undersampling-nearmiss.

## Pipeline por fold

```
datos crudos
  -> load_and_clean (limpia + binariza target)
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

### Umbral de decision (`--threshold-metric`)

El corte de 0.5 se reemplaza por el umbral que maximiza una metrica,
elegido por modelo sobre predicciones **out-of-fold** (sin leakage de
test). Opciones: `f1` (default), `youden`, `cost`. Para `cost`, `--cost-ratio`
pesa FN vs FP (default 1.0). ROC-AUC/PR-AUC/Brier no cambian (usan
probabilidades). Ver `reports/changes_and_results.md` para los deltas
observados: mejora recall/F1 en los modelos conservadores en 0.5 sin
afectar las metricas de probabilidad.

### Sin tuning de hiperparametros

Se probo tuning por grilla (GridSearchCV, PR-AUC) y se descarto: la
ganancia sobre el threshold tuning ya presente es marginal (f1 ~+0.001 a
+0.01, ROC-AUC/PR-AUC planos) y no justifica el mecanismo extra. Los
defaults de sklearn/XGBoost/LightGBM ya estan cerca del optimo. El lift
medido viene del feature set y la estrategia de balanceo, no del tuning.
Ver `reports/changes_and_results.md` (resumen final) para el detalle.

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

# Iteracion rapida con subsample pequeno
python brfss_dataset_model_ADDEPEV3.py --subsample 30000 --cv 2

# Feature set: unico preset 'default' (no hace falta elegir)
python brfss_dataset_model.py --feature-set default --cv 3

# Full data (sin --subsample) — ~30 min con --balancing class_weight
python brfss_dataset_model_ADDEPEV3.py --cv 5

# Probar otra estrategia de balanceo (smote, undersampling-random, nearmiss)
python brfss_dataset_model_ADDEPEV3.py --balancing undersampling-nearmiss --subsample 100000 --cv 5
```

### Recomendaciones (de los benchmarks en `reports/`)

- **Mejor combinacion por defecto**: `ADDEPEV3` + `default` +
  `--balancing class_weight` (`_RACE` ya incluida por defecto). Reproducir con:
  ```bash
  python brfss_dataset_model_ADDEPEV3.py --subsample 100000 --cv 5
  ```
  Espera ~2 min en CPU moderna. RandomForest lidera con ROC-AUC ~0.83 y
  F1 ~0.59 en el set `default`.

- `ADDEPEV3` es un target dificil por base rate 21% y senal historica:
  recall ~0.40 con todos los modelos; ROC-AUC ~0.83 con el set `default`
  + class_weight.

- **Smoke test rapido** (~30 s): agregar `--subsample 30000 --cv 2`.

- **Sin tuning de hiperparametros**: se probo y se descarto (ver seccion
  anterior). Los defaults funcionan bien; el lift medido viene del feature
  set y la estrategia de balanceo, no del tuning.

Salidas en `metadata/model_results/`:

- `folds_<target>_<feature_set>_<balancing>.csv` — una fila por (modelo, fold, metrica).
- `summary_<target>_<feature_set>_<balancing>.csv` — mean y std por (modelo, metrica).
- `cm_<modelo>_<target>_<feature_set>_<balancing>.png` — matriz de confusion acumulada.

## Resultados

El registro de cambios y el reporte final (full data, 5-fold CV) viven en
`reports/changes_and_results.md`: metricas por modelo, veredicto por
cambio (mantenidos y descartados) y configuracion recomendada.

Metrica principal: **ROC-AUC**, complementada con **PR-AUC** y **Brier**.

## Estructura del repo

```
brfss_dataset_model.py                # libreria: logica del pipeline + CLI
brfss_dataset_model_ADDEPEV3.py       # wrapper: target=ADDEPEV3, set=default
tools/
  select_rf_top30.py                  # seleccion del set default (RF + MI + permutation)
  export_columns.py                   # exporta metadata de columnas a CSV/MD
requirements.txt                      # dependencias Python
metadata/
  columns/
    brfss_2024_columns.csv            # 301 filas: position, name, label
    brfss_2024_columns.md
  model_results/                      # salidas del pipeline (folds, summary, CM)
reports/
  changes_and_results.md              # registro de cambios + reporte final (5-fold)
legacy/                               # archivos de versiones anteriores (historico)
```

## Seleccion de features y target

El set `default` se produce con `tools/select_rf_top30.py` sobre el
universo completo de 301 columnas (menos ID/geo/peso y el target):
RandomForest importance + informacion mutua + permutation importance como
contraste. El canonico es el top-30 de RF. Ver `reports/changes_and_results.md`
para el detalle.

El target es `ADDEPEV3` (diagnostico de depresion de por vida), la unica
variable de diagnostico de depresion explicita en las 301 columnas. La
auditoria historica de 4 targets candidatos vive en `legacy/`.

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
- Sin tuning de hiperparametros (se probo y se descarto; ver
  `reports/changes_and_results.md`). El lift medido en los benchmarks
  viene del feature set y la estrategia de balanceo, no del tuning.
- `ADDEPEV3` (depresion de por vida) es un target dificil: base rate
  21%, recall ~0.40 con todos los modelos, AUC ~0.83 con el set `default`
  + class_weight. El "always no" baseline da 79% accuracy, pero eso
  enmascara el problema (recall ~0).
  Para screening, el recall es la metrica relevante y sigue siendo bajo.
- `_RACE` se incluye por defecto como mean-encoded (no one-hot) para
  preservar el sample size.
- Threshold fijo en 0.5. Para `ADDEPEV3` ajustarlo puede mejorar recall
  a costa de precision, pero el limite real es la senal insuficiente.
