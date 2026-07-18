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
aparta como columna mean-encoded (no cuenta dentro de las 30). Las
variables ACE (childhood adversity) estaban en el pool de candidatos pero
ninguna entro al top-30 del RF — probablemente por su alta missingness
(modulo opcional en BRFSS, muchos estados no lo administran). No se usa
`ace_score` ni los items ACE individuales en el preset `default`.

| Preset  | Total predictores | Numericas | Ordinales | Categoricas | `_RACE` |
|---------|-------------------|-----------|-----------|-------------|---------|
| default | 30                | 8         | 4         | 18          | si (mean-encoded) |

El set canonico da ROC-AUC ~0.833 en hold-out (RF, 100k), casi igual al
universo completo de 293 columnas (~0.840). Las variables de sintoma de
salud (`MENTHLTH`, `POORHLTH`, `_PHYS14D`, `_RFHLTH`) estan incluidas: no
son leakage del target (diagnostico de por vida) pero inflan el
performance por ser del mismo constructo.

### Las 30 variables del preset `default`

Numericas (8): `_BMI5`, `_AGE80`, `PHYSHLTH`, `MENTHLTH`, `POORHLTH`,
`HEIGHT3`, `WTKG3`, `_PACKYRS`.

Ordinales (4): `_AGEG5YR`, `GENHLTH`, `SDLONELY`, `LSATISFY`.

Categoricas (18): `DECIDE`, `DIFFALON`, `_PHYS14D`, `_RFHLTH`, `SEXVAR`,
`_SEX`, `ECIGNOW3`, `EMPLOY1`, `ASTHMA3`, `_AIDTST4`, `_LTASTH1`,
`HAVARTH4`, `HIVTST7`, `CELLSEX3`, `HIVTSTD3`, `_ASTHMS1`, `MEDCOST1`,
`_DRDXAR2`.

Mean-encoded (extra, fuera del conteo de 30): `_RACE`.

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
  python brfss_dataset_model_ADDEPEV3.py --cv 5
  ```
  Full data 5-fold: ~32 min en CPU moderna. XGBoost y LightGBM lideran
  ROC-AUC (~0.843) y PR-AUC (0.635); XGBoost tiene el mejor Brier (0.116).
  LightGBM lidera F1 al umbral optimo (~0.597). Ver `reports/final_results.md`
  para la tabla completa.

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

El reporte final (full data, 5-fold CV, metricas por modelo y configuracion
recomendada) vive en `reports/final_results.md`. El registro completo de
cambios — que se probo, que se mantuvo y que se descarto — vive en
`reports/changes_and_results.md`.

Metrica principal: **ROC-AUC**, complementada con **PR-AUC** y **Brier**.

## Decisiones de diseno

Decisiones tomadas para mejorar las metricas del pipeline. Cada entrada
dice que es, por que ayuda (o por que no), y el impacto medido en full
data (455k, 5-fold CV). Detalle completo del proceso en
`reports/changes_and_results.md`; numeros finales en `reports/final_results.md`.

### Mantenidas

**Preset unico `default` desde el universo completo**
- Que es: un solo feature set de 30 predictores, seleccionados de las 301
  columnas del BRFSS (menos ID/geo/peso y el target) con RandomForest
  importance (+ informacion mutua + permutation como contraste). Reemplaza
  los presets `recommended`/`rf_top30` y elimina la logica de `use_ace`/
  `ace_score`.
- Por que ayuda: los presets anteriores usaban un pool curado a mano de
  ~50 columnas. Escanear el universo completo deja que el RF encuentre
  predictores que la curacion manual dejo fuera (ej. `_AGE80`, `HEIGHT3`,
  `WTKG3`, `_PACKYRS`, `SEXVAR`). RF sobre 293 columnas da ROC-AUC 0.8405;
  el top-30 canonico da 0.8332 — solo 0.7 pts por debajo del universo, asi
  que las 30 columnas capturan casi toda la senal disponible.
- Impacto: ROC-AUC +0.036–0.055 y PR-AUC +0.057–0.086 vs el `recommended`
  original, en todos los modelos. **El mayor lift de todo el proyecto.**

**Threshold tuning (f1 OOF)**
- Que es: por cada modelo, elegir el umbral de decision que maximiza F1
  sobre las predicciones out-of-fold (OOF) del CV, en vez de usar 0.5.
  Flag `--threshold-metric {f1,youden,cost}`.
- Por que ayuda: los modelos conservadores (LinearSVC, XGBoost) operan a
  0.5 con recall muy bajo (0.38, 0.44) porque su umbral por defecto
  favorece la specificity. Elegir el umbral sobre OOF (no sobre test, sin
  leakage) mueve el punto de operacion hacia donde el modelo tiene mejor
  F1. ROC-AUC/PR-AUC/Brier no cambian (usan probabilidades, no el corte),
  asi que no es cheating.
- Impacto: F1 0.49–0.58 → 0.58–0.60; recall 0.38–0.75 → 0.64–0.65.
  **Tradeoff**: precision baja de ~0.69 a ~0.53 y specificity de ~0.95 a
  ~0.85 en los modelos conservadores. Es el costo esperado de mover el
  umbral, no un bug.

### Descartadas (probadas, sin mejora razonable)

**HP tuning (GridSearchCV)**
- Que es: busqueda en grilla por modelo (class_weight multiplier +
  complejidad/regularizacion), scoring PR-AUC, sobre un subsample con CV
  interno. Escribia `hp_best.json` que `main()` leia para hacer
  `set_params`.
- Por que no ayuda: el threshold tuning ya optimiza el punto de operacion;
  el multiplicador de class_weight que el grid elige (1:3, 1:5) desplaza
  el umbral de decision, no el ranking — y el ranking es lo unico que
  mueve ROC-AUC/PR-AUC. Los defaults ya estaban cerca del optimo.
- Resultado: marginal (f1 ~+0.001–0.01, ROC-AUC/PR-AUC planos). Borrado.

**Interacciones degree-2 (modelos lineales)**
- Que es: encoder alternativo con `PolynomialFeatures(degree=2,
  interaction_only=True)` sobre el bloque numerico, aplicado solo a
  LinearSVC y LogReg. Darles las interacciones numerico×numerico que los
  arboles capturan nativamente.
- Por que no ayuda: la hipotesis era que el techo bajo de los lineales
  venia de falta de no-linealidad. No es asi: la senal es ya cercana a
  lineal, o los 78 terminos extra sobreajustan (fold 3 empeora). El techo
  de los lineales esta fijado por otro lado (capacidad lineal sobre estas
  30 features), no por interacciones faltantes.
- Resultado: nulo (ROC-AUC plano, f1 sin mover). Revertido.

**Ensemble (StackingClassifier)**
- Que es: sexto modelo, stacking de las 5 bases con un meta-learner
  LogisticRegression y CV interno (cv=3) para generar predicciones OOF de
  las bases como features del meta-learner, sin leakage.
- Por que no ayuda (en full data): en subsamples de 30k el Ensemble ganaba
  ~+0.01 ROC-AUC sobre la mejor base — las bases no habian llegado al
  techo. En 455k las bases (XGBoost/LightGBM) ya llegan al mismo techo
  (0.843), asi que el stacking no suma. Ademas es un modelo aparte: no
  levanta las metricas peores (LinearSVC/XGBoost siguen bajos), solo
  agrega una fila mas que empata a las mejores. Costo: ~3-4x el fit por
  fold.
- Resultado: empata a XGBoost/LightGBM (ROC-AUC 0.843, F1@thr 0.597) sin
  superarlos y no afecta las metricas peores. Borrado.

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
  feature_selection/
    default_selection.md              # reporte de seleccion del set default
  model_results/                      # salidas del pipeline (folds, summary, CM)
reports/
  final_results.md                   # reporte final (5-fold full data, metricas + config)
  changes_and_results.md              # registro de cambios (kept/discarded audit trail)
legacy/                               # archivos de versiones anteriores (historico)
```

## Seleccion de features y target

El set `default` se produce con `tools/select_rf_top30.py` sobre el
universo completo de 301 columnas (menos ID/geo/peso/estructurales y el
target): RandomForest importance + informacion mutua + permutation
importance como contraste. El canonico es el top-30 de RF. El reporte de
seleccion (rankings, scores, overlap) se escribe en
`metadata/feature_selection/default_selection.md`; ver tambien
`reports/changes_and_results.md` para el detalle del proceso.

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

- Full data (455k filas) con `--balancing class_weight` (default) y 5-fold
  CV corre en ~32 min; con `smote` o `undersampling-nearmiss` seria ~5-7x
  mas lento por el KNN del resampleo. Para iteracion rapida, usar
  `--subsample` (ej. `30000 --cv 2`, ~30 s).
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
- El umbral por defecto es F1-optimo por modelo sobre OOF
  (`--threshold-metric f1`), no 0.5. Operar en 0.5 daria recall mas bajo
  en los modelos conservadores (LinearSVC ~0.38, XGBoost ~0.44); el
  tuning lo sube a ~0.64 a costa de precision. El limite real sigue
  siendo la senal insuficiente.
