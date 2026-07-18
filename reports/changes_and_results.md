# Cambios y resultados — brfss_dataset_model

Registro incremental de cambios al pipeline y sus resultados empiricos.
Las entradas mas recientes van arriba. Para cambios futuros, agregar una
nueva seccion `## ` con fecha y mantener el formato.

---

## 2026-07-18 — Reporte final (full data, 455k, 5-fold CV)

Pipeline final tras mantener solo los cambios que aportaron mejora
razonable. Corrida autoritativa sobre el dataset completo (455,006 filas,
5-fold CV estratificado, `--balancing class_weight`, `--threshold-metric
f1`). Feature set unico `default` (30 predictores + `_RACE` mean-encoded).
5 modelos base (sin Ensemble; ver descartados).

### Metricas por modelo (mean sobre 5 folds)

**ROC-AUC / PR-AUC / Brier** (metricas de probabilidad, independientes del
umbral):

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| RandomForest      | 0.8312 | 0.6118 | 0.1709 |
| LinearSVC         | 0.8253 | 0.6071 | 0.1213 |
| LogisticRegression| 0.8260 | 0.6079 | 0.1636 |
| XGBoost           | **0.8433** | **0.6354** | **0.1156** |
| LightGBM          | 0.8431 | 0.6354 | 0.1568 |

**F1 / Recall / Precision / Specificity** a umbral 0.5 (operacion por
defecto del modelo, sin tuning):

| model | F1 | Recall | Precision | Specificity |
|---|---|---|---|---|
| RandomForest      | 0.5710 | 0.7285 | 0.4695 | 0.7795 |
| LinearSVC         | 0.4893 | 0.3781 | 0.6935 | 0.9552 |
| LogisticRegression| 0.5733 | 0.6905 | 0.4901 | 0.8075 |
| XGBoost           | 0.5383 | 0.4408 | 0.6913 | 0.9473 |
| LightGBM          | 0.5849 | 0.7497 | 0.4795 | 0.7820 |

**Mismas metricas al umbral F1-optimo OOF** (threshold tuning, sin
leakage):

| model | F1@thr | Recall@thr | Precision@thr | Specificity@thr | umbral |
|---|---|---|---|---|---|
| RandomForest      | 0.5812 | 0.6463 | 0.5279 | 0.8452 | 0.553 |
| LinearSVC         | 0.5768 | 0.6381 | 0.5263 | 0.8462 | 0.248 |
| LogisticRegression| 0.5778 | 0.6457 | 0.5228 | 0.8421 | 0.545 |
| XGBoost           | 0.5972 | 0.6445 | 0.5564 | 0.8624 | 0.304 |
| LightGBM          | **0.5974** | 0.6517 | 0.5515 | 0.8581 | 0.614 |

### Lectura
- **XGBoost y LightGBM empatan en el techo de ranking** (ROC-AUC ~0.843,
  PR-AUC 0.6354). XGBoost ademas tiene el mejor Brier (0.1156) —
  probabilidades mejor calibradas.
- **Umbral 0.5 vs umbral F1-optimo**: el threshold tuning sube F1 de
  ~0.49-0.58 a ~0.58-0.60 en todos los modelos. LinearSVC y XGBoost
  (conservadores en 0.5, recall 0.38/0.44) suben su recall a ~0.64 al
  bajar su umbral (0.248 / 0.304). Costo: precision baja de ~0.69 a
  ~0.53-0.56 y specificity de ~0.95 a ~0.85-0.86 en esos dos. Tradeoff
  esperado y documentado.
- **Estabilidad**: std entre folds ~0.000-0.007 — resultados muy estables
  en full data.

### Veredicto por cambio (mantenidos)

| cambio | impacto en full data | veredicto |
|---|---|---|
| **`default` preset (universo)** | ROC-AUC ~0.83-0.84, PR-AUC ~0.61-0.64. El techo actual. | **Vale la pena.** Mayor impacto, simplifica el pipeline. |
| **Threshold tuning (f1 OOF)** | F1 0.49-0.58 -> 0.58-0.60; recall 0.38-0.75 -> 0.64-0.65. Tradeoff: precision/specificity bajan en los conservadores. | **Vale la pena** para screening. No es gratis. |

### Veredicto por cambio (descartados)

| cambio | resultado | veredicto |
|---|---|---|
| **HP tuning (GridSearchCV)** | Marginal (f1 ~+0.001 a +0.01, ROC-AUC plano). | **Descartado.** Defaults ya cerca del optimo. Borrado. |
| **Interacciones degree-2 (lineales)** | Nulo (ROC-AUC plano, f1 sin mover). | **Descartado.** El techo de los lineales no es falta de interacciones. Revertido. |
| **Ensemble (StackingClassifier)** | En full data empata a XGBoost/LightGBM (ROC-AUC 0.843, F1@thr 0.597) sin superarlos y no afecta las metricas peores (LinearSVC/XGBoost siguen bajos). Costo ~3-4x el fit. | **Descartado.** En subsamples de 30k ganaba ~+0.01 ROC-AUC; en 455k las bases ya llegan al mismo techo. No aporta valor al set de modelos. Borrado. |

### Configuracion final recomendada
- Feature set: `default` (unico).
- Balanceo: `--balancing class_weight`.
- Umbral: `--threshold-metric f1` (default) para screening; considerar
  `--threshold-metric cost --cost-ratio 3` si se quiere priorizar recall
  todavia mas (no probado en full data, pero el mecanismo esta).
- Modelo: **XGBoost** si se quiere mejor calibracion (Brier 0.116) y el
  techo de ROC-AUC/PR-AUC; **LightGBM** si se quiere el mismo techo con
  recall ligeramente mayor al umbral optimo; **LinearSVC** si se quiere
  maxima specificity a costa de recall (operando en umbral 0.5).

---

## 2026-07-16 — Resumen final: cambios mantenidos y descartados

Tras probar varias direcciones para mejorar las metricas, este es el
balance final. El pipeline actual conserva solo los cambios que aportaron
mejora razonable; los descartados se probaron, se midieron y se borraron
(esta seccion los documenta para que no se repitan).

### Mantenidos (aportan mejora)

| cambio | que hace | impacto medido (30k/3-fold, vs original) |
|---|---|---|
| **Unico preset `default` desde el universo completo** | Selecciona 30 predictores de las 301 columnas via RF importance (+ MI + permutation como contraste). Reemplaza `recommended`/`rf_top30` y elimina `use_ace`/`ace_score`/`ACE_VARIABLES`. | ROC-AUC +0.036 a +0.055 en todos los modelos; PR-AUC +0.057 a +0.086; Brier −0.01 a −0.03. **El mayor lift.** |
| **Threshold tuning (f1 OOF)** | Elige el umbral de decision por modelo sobre predicciones out-of-fold (no leakage). Flag `--threshold-metric {f1,youden,cost}`. | F1 +0.05 a +0.16 y recall +0.04 a +0.36 en los modelos conservadores. **Tradeoff**: precision/specificity bajan en esos mismos. ROC-AUC/PR-AUC/Brier no cambian. |

### Descartados (se probaron, no aportaron mejora razonable)

| cambio | resultado | por que se descarta |
|---|---|---|
| **Tuning de hiperparametros (GridSearchCV, `hp_best.json`)** | Marginal: f1 ~+0.001 a +0.01 sobre el threshold tuning, ROC-AUC/PR-AUC planos. | Los defaults ya estaban cerca del optimo; el mecanismo completo (script + JSON + wiring en `build_models`/`main`) agrega complejidad sin cambiar los numeros. Borrado. |
| **Interacciones degree-2 para modelos lineales** | Nulo: ROC-AUC plano (LinearSVC 0.831->0.833/0.831/0.828; LogReg 0.831->0.832/0.831/0.827), f1 sin moverse. | El techo de los lineales no esta fijado por falta de interacciones. Revertido. |
| **Ensemble (StackingClassifier + LogReg)** | En subsamples de 30k ganaba ~+0.01 ROC-AUC sobre la mejor base; en full data (455k, 5-fold) empata a XGBoost/LightGBM (ROC-AUC 0.843, F1@thr 0.597) sin superarlos y no afecta las metricas peores (LinearSVC/XGBoost siguen bajos). | No aporta valor al set de modelos en full data y cuesta ~3-4x el fit por fold. Borrado. |

El detalle de cada cambio vive en las secciones siguientes (de arriba
abajo: veredicto global con tablas antes/despues, interacciones revertidas,
Ensemble, preset unico, tuning descartado).

---

## 2026-07-16 — Veredicto global: original vs actual (mismos datos)

Comparativo honesto del pipeline **original** (feature set `recommended`,
umbral 0.5 fijo, sin HP, sin Ensemble, 5 modelos base) contra el
**actual** (`default`, threshold tuning f1 OOF, HP on, Ensemble on, 6
modelos). Mismos datos para ambos: ADDEPEV3, 30k subsample, 3-fold CV,
`--balancing class_weight`, seed 42.

Notese: el `recommended` original incluia `ace_score`; como `add_ace_score`
se borro al eliminar ACE, el comparativo usa `recommended` sin `ace_score`
(el techo lineal del original). Es una pequeña concesion contra el
original real, conservadora (favorece al original).

### ROC-AUC
| model | A (original) | B (actual) | delta |
|---|---|---|---|
| RandomForest      | 0.7943 | 0.8372 | +0.0429 |
| LinearSVC         | 0.7941 | 0.8306 | +0.0364 |
| LogisticRegression| 0.7945 | 0.8307 | +0.0363 |
| XGBoost           | 0.7840 | 0.8389 | +0.0550 |
| LightGBM          | 0.7840 | 0.8357 | +0.0517 |
| Ensemble          |   —    | 0.8423 |   n/a   |

### PR-AUC
| model | A | B | delta |
|---|---|---|---|
| RandomForest      | 0.5563 | 0.6192 | +0.0630 |
| LinearSVC         | 0.5539 | 0.6115 | +0.0576 |
| LogisticRegression| 0.5543 | 0.6116 | +0.0573 |
| XGBoost           | 0.5376 | 0.6236 | +0.0861 |
| LightGBM          | 0.5426 | 0.6217 | +0.0791 |
| Ensemble          |   —    | 0.6294 |   n/a   |

### Brier (mas bajo = mejor)
| model | A | B | delta |
|---|---|---|---|
| RandomForest      | 0.1688 | 0.1361 | −0.0327 |
| LinearSVC         | 0.1298 | 0.1202 | −0.0095 |
| LogisticRegression| 0.1797 | 0.1611 | −0.0186 |
| XGBoost           | 0.1337 | 0.1175 | −0.0163 |
| LightGBM          | 0.1636 | 0.1403 | −0.0233 |
| Ensemble          |   —    | 0.1597 |   n/a   |

### F1 (umbral 0.5 en A, umbral F1-optimo OOF en B)
| model | A | B | delta |
|---|---|---|---|
| RandomForest      | 0.5412 | 0.5883 | +0.0471 |
| LinearSVC         | 0.4188 | 0.5831 | +0.1643 |
| LogisticRegression| 0.5270 | 0.5825 | +0.0555 |
| XGBoost           | 0.4489 | 0.5916 | +0.1428 |
| LightGBM          | 0.5270 | 0.5878 | +0.0609 |
| Ensemble          |   —    | 0.5911 |   n/a   |

### Recall (umbral 0.5 en A, umbral F1-optimo en B)
| model | A | B | delta |
|---|---|---|---|
| RandomForest      | 0.6183 | 0.6630 | +0.0447 |
| LinearSVC         | 0.3034 | 0.6617 | +0.3583 |
| LogisticRegression| 0.6693 | 0.6478 | −0.0215 |
| XGBoost           | 0.3522 | 0.6308 | +0.2786 |
| LightGBM          | 0.6196 | 0.6049 | −0.0147 |
| Ensemble          |   —    | 0.6183 |   n/a   |

### Precision (umbral 0.5 en A, umbral F1-optimo en B)
| model | A | B | delta |
|---|---|---|---|
| RandomForest      | 0.4813 | 0.5287 | +0.0474 |
| LinearSVC         | 0.6763 | 0.5212 | −0.1551 |
| LogisticRegression| 0.4346 | 0.5291 | +0.0945 |
| XGBoost           | 0.6190 | 0.5571 | −0.0620 |
| LightGBM          | 0.4584 | 0.5716 | +0.1132 |
| Ensemble          |   —    | 0.5661 |   n/a   |

### Specificity
| model | A | B | delta |
|---|---|---|---|
| RandomForest      | 0.8212 | 0.8417 | +0.0205 |
| LinearSVC         | 0.9611 | 0.8372 | −0.1240 |
| LogisticRegression| 0.7667 | 0.8455 | +0.0788 |
| XGBoost           | 0.9419 | 0.8656 | −0.0763 |
| LightGBM          | 0.8039 | 0.8786 | +0.0747 |
| Ensemble          |   —    | 0.8730 |   n/a   |

### Veredicto por cambio

> Nota: el Ensemble fue descartado despues (ver el reporte final y el
> resumen al inicio de este archivo). La tabla abajo refleja el veredicto
> previo en 30k, antes de la corrida en full data.

| cambio | impacto metrico | veredicto |
|---|---|---|
| **Single `default` preset (universo)** | ROC-AUC +0.036–0.055, PR-AUC +0.057–0.086, Brier −0.01 a −0.03 **en todos los modelos**. La mayor parte del lift viene de aca (mejores features). | **Vale la pena.** Es el cambio de mayor impacto y simplifica el pipeline (un solo preset en vez de dos). |
| **Threshold tuning (f1 OOF)** | F1 +0.05 a +0.16, Recall +0.04 a +0.36 en los modelos conservadores (LinearSVC/XGBoost). Costo: precision −0.06 a −0.16 y specificity −0.08 a −0.12 en esos mismos (tradeoff esperado al mover el umbral). Modelos ya balanceados (RF/LogReg/LightGBM) apenas se mueven. ROC-AUC/PR-AUC/Brier no cambian (no hay leakage). | **Vale la pena** para screening (recall importa mas). No es gratis: es un tradeoff, no una mejora gratis. Para uso clinico donde precision manda, dejar `--threshold-metric` en un valor que no baje precision tanto. |
| **HP tuning (GridSearchCV, hp_best.json)** | Marginal sobre el threshold tuning (medido: f1 ~+0.001 a +0.01). ROC-AUC/PR-AUC planos. | **Marginal.** Mantener el mecanismo (off por defecto) pero no cambia los numeros reportados. Confirmo que los defaults ya estaban cerca del optimo. |
| **Ensemble (StackingClassifier + LogReg)** | ROC-AUC 0.8423 — el mas alto de todos, +0.003 a +0.015 sobre la mejor base (XGBoost 0.8389). F1 0.5911 (top). Es el unico cambio que levanta el **piso de ranking** sin tradeoff precision/recall. | **Vale la pena.** Es el levantamiento estructural real. Costo: ~3-4x el fit de las bases por fold. Skipped: meta-learner GBM (margen marginal sobre LogReg). |
| **Interacciones degree-2 (revertido)** | Nulo (ROC-AUC plano, f1 sin mover). | **No vale la pena.** Revertido. Las interacciones no atacan el techo real de los lineales. |

### Sintesis
- El lift total de ROC-AUC (~+0.04) y PR-AUC (~+0.06) viene **casi todo del
  feature set `default`** (mejores predictores desde el universo completo).
- El lift de F1/Recall en los modelos conservadores viene **del threshold
  tuning** (tradeoff contra precision/specificity, no es gratis).
- El **Ensemble** es la unica mejora que sube el ranking **sin tradeoff**;
  pequeña en magnitud (+0.005–0.015 ROC-AUC) pero consistente.
- El **HP tuning** y las **interacciones** son marginales/nulos; el primero se
  mantiene como scaffolding opcional, el segundo se borro.

---

## 2026-07-16 — Interacciones degree-2 para modelos lineales (revertido)

### Cambio probado
Se agrego un encoder alternativo (`make_encoder_lin`) con
`PolynomialFeatures(degree=2, interaction_only=True)` sobre el bloque
numerico, aplicado solo a LinearSVC y LogisticRegression. Objetivo: darles
el no-linealidad que los arboles capturan nativamente, para levantar su
ROC-AUC (y asi su recall/f1 a precision igual, sin tradeoff).

### Resultados (ADDEPEV3, `default`, 30k subsample, 3-fold)
ROC-AUC antes vs despues de interacciones:

| modelo         | antes (3 folds)      | despues             | delta        |
|----------------|---------------------|---------------------|--------------|
| LinearSVC      | 0.831/0.830/0.830   | 0.833/0.831/0.828   | ~plano (-0.01 a +0.003) |
| LogisticReg    | 0.831/0.831/0.830   | 0.832/0.831/0.827   | ~plano (-0.003 a +0.001) |

f1 y recall en el umbral F1-optimo tampoco se movieron (LinearSVC ~0.50,
LogReg ~0.58). Arboles y Ensemble sin cambios (sanity: el ruteo funciono).

### Conclusion
Interacciones degree-2 no ayudan. El techo de los modelos lineales no esta
fijado por falta de interacciones — la senal ya es cercana a lineal, o los
78 terminos extra sobreajustan (fold 3 empeora). Se revierte el cambio
completo (`make_encoder_lin`, el ruteo en el loop, los imports de
`Pipeline`/`PolynomialFeatures`). Conclusion honesta: el stacking (entrada
anterior) sigue siendo el unico levantamiento real del piso de ranking; las
interacciones no suman.

---

## 2026-07-16 — Ensemble por stacking (meta-learner LogReg) — DESCARTADO

> **Descartado.** En subsamples de 30k el Ensemble ganaba ~+0.01 ROC-AUC
> sobre la mejor base; en full data (455k, 5-fold) **empata** a
> XGBoost/LightGBM (ROC-AUC 0.843, F1@thr 0.597) sin superarlos, y **no
> afecta las metricas peores** (LinearSVC y XGBoost siguen siendo los
> mas bajos). No aporta valor al set de modelos y cuesta ~3-4x el fit
> por fold. Se borra `StackingClassifier` de `build_models`. Ver el
> resumen final arriba.

### Cambio
Se agrego un sexto modelo, **`Ensemble`**, construido con
`StackingClassifier` de sklearn sobre los 5 modelos base, con
`final_estimator=LogisticRegression` (meta-learner). El stacker hace CV
interno (`cv=3`) sobre el train de cada fold para generar predicciones
out-of-fold de las bases como features del meta-learner -> sin leakage de
test. `stack_method="predict_proba"`: las features del meta-learner son las
probabilidades de las 5 bases (5 dimensiones).

Motivo: las metricas mas bajas eran recall + f1 en los modelos
conservadores (LinearSVC, LightGBM) a su umbral F1-optimo, mientras su
precision ya era alta. Un movimiento de umbral/costo solo cambiaria
precision por recall (tradeoff). El stacking levanta el piso de ranking
(ROC-AUC), lo que deja mas recall disponible a precision igual; no es un
tradeoff. Las bases del stacker usan los `hp` tuned de `hp_best.json`; el
meta-learner usa defaults.

Costo: ~3-4x el fit de las bases por fold (CV interno del stacker).

### Resultados (ADDEPEV3, `default`, 30k subsample, 3-fold, class_weight, hp on)
ROC-AUC por fold (Ensemble vs mejores bases):

| fold | RandomForest | LinearSVC | LogReg | XGBoost | LightGBM | **Ensemble** |
|------|--------------|-----------|--------|---------|----------|--------------|
| 1    | 0.841        | 0.831     | 0.831  | 0.843   | 0.840    | **0.846**    |
| 2    | 0.838        | 0.830     | 0.831  | 0.842   | 0.839    | **0.842**    |
| 3    | 0.833        | 0.830     | 0.830  | 0.832   | 0.829    | **0.839**    |

Ensemble es el modelo de mayor ROC-AUC en los 3 folds (~+0.005 a +0.015
sobre la mejor base). f1 por fold ~0.58-0.59, en el topo junto a
RandomForest y LightGBM; supera a LinearSVC (~0.50) y XGBoost (~0.53).

### Conclusion
El stacking levanta el techo de ranking sin tocar el umbral ni el balanceo,
que era el objetivo (mejorar recall/f1 sin tradeoff de precision). Se
mantiene como un modelo mas del pipeline (reporta las mismas metricas,
umbral OOF y CM que los otros). Skipped: meta-learner con gradient boosting
(margen marginal sobre LogReg, mas lento).

---

## 2026-07-16 — Unico preset `default` desde el universo completo

### Cambio
Se eliminaron los presets `recommended` y `rf_top30` y el flag/logica
`use_ace` / `ace_score` / `ACE_VARIABLES`. Ahora hay un unico preset
`default`: 30 predictores seleccionados sobre **todas las 301 columnas**
(menos ID/geo/peso y el target `ADDEPEV3`) con `tools/select_rf_top30.py`,
que corre RF importance + mutual information + permutation importance
como contraste.

- Excluidos del canonico RF-top-30: `_MENT14D` (casi-duplicado del target)
  y `_STSTR` (estrato de muestreo). Agregados `_PACKYRS` y `MEDCOST1` del
  rango RF siguiente. `_RACE` se mean-encodea aparte (no cuenta en las 30).
- La seleccion usa label-encoding de todas las columnas (BRFSS es codigo
  de categoria); MI sobre 5k filas y permutation sobre el top-30 (re-entre-
  nado en 30 cols) para no explotar memoria.
- `hp_best.json` ahora se keyea por `"default"`; el tuner se re-corrio para
  ese set.

### Resultados (ADDEPEV3, hold-out del selector, 100k)
- RF sobre 293 cols (universo): test ROC-AUC 0.8405.
- RF sobre el top-30 canonico: test ROC-AUC 0.8332 (solo -0.7 pts).
- Overlap RF ∩ MI (top-30): 21/30. MI contamina con pesos de diseno
  (`_LLCPWT*`, `WEIGHT2`, `IDATE`) — por eso el canonico es RF, no MI.
- Pipeline (default, 10k/3-fold, class_weight): ROC-AUC ~0.81–0.84 por
  modelo; RandomForest ~0.836, LogReg ~0.829.

### Conclusion
Un solo preset, seleccionado sobre el universo completo, es mas simple y
casi tan bueno como el universo entero. `recommended`/`rf_top30` y ACE
eran redundantes; se borraron. Documentado en README.



### Cambio
Se agrego seleccion de umbral de decision por modelo sobre predicciones
**out-of-fold** (sin leakage de test):

- Nuevo flag `--threshold-metric {f1, youden, cost}` (default `f1`).
- Nuevo flag `--cost-ratio` (default 1.0) para el modo `cost`.
- En `brfss_dataset_model.py`: se acumulan `(y_true, y_proba)` por modelo
  por fold; tras la CV se elige el umbral que maximiza la metrica pedida
  con `precision_recall_curve`, y se recalculan precision/recall/f1/
  accuracy/specificity en ese umbral (`*@thr` en `folds_*.csv` y
  `summary_*.csv`, mas fila `threshold`). El CM PNG se grafica en el
  umbral elegido.

El umbral se elige SOLO de OOF; ROC-AUC, PR-AUC y Brier usan probabilidades
y no cambian. No es cheating: no altera features, target ni entrenamiento.

### Resultados (ADDEPEV3, 10k subsample, 3-fold, `--balancing class_weight`)
Delta = metrica @ umbral optimo − metrica @ 0.5 (signo + = mejora).

**rf_top30**

| metric | LightGBM | LinearSVC | LogReg | RandomForest | XGBoost |
|---|---|---|---|---|---|
| accuracy | −0.030 | −0.033 | +0.018 | +0.001 | −0.026 |
| precision | −0.056 | −0.171 | +0.032 | +0.000 | −0.102 |
| recall | +0.097 | +0.276 | −0.042 | −0.000 | +0.154 |
| f1 | +0.004 | +0.091 | +0.006 | +0.001 | +0.040 |
| specificity | −0.065 | −0.116 | +0.034 | +0.001 | −0.075 |
| roc_auc / pr_auc / brier | n/a | n/a | n/a | n/a | n/a |

**recommended**

| metric | LightGBM | LinearSVC | LogReg | RandomForest | XGBoost |
|---|---|---|---|---|---|
| accuracy | −0.021 | −0.051 | +0.030 | −0.014 | −0.055 |
| precision | −0.031 | −0.214 | +0.038 | −0.025 | −0.151 |
| recall | +0.059 | +0.312 | −0.061 | +0.039 | +0.228 |
| f1 | +0.006 | +0.123 | +0.006 | +0.002 | +0.066 |
| specificity | −0.042 | −0.148 | +0.055 | −0.028 | −0.130 |
| roc_auc / pr_auc / brier | n/a | n/a | n/a | n/a | n/a |

Umbrales F1-optimos (rf_top30): RandomForest 0.501, LinearSVC 0.241,
LogisticRegression 0.545, XGBoost 0.280, LightGBM 0.367.

### Conclusion
Se mantiene. Todo modelo mejora o mantiene F1; los modelos conservadores
en 0.5 (LinearSVC, XGBoost, LightGBM) ganan recall +0.15–0.31. El costo
es precision menor en esos modelos (tradeoff esperado). ROC-AUC/PR-AUC/
Brier identicos al baseline -> sin leakage.

---

## 2026-07-15 — Tuning de hiperparametros por grilla (GridSearchCV) — DESCARTADO

> **Descartado.** Se probo, la ganancia fue marginal (f1 ~+0.001 a +0.01
> sobre el threshold tuning, ROC-AUC/PR-AUC planos) y no justifica el
> mecanismo (script + JSON + wiring). Se borro `tools/tune_hparams.py` y
> `metadata/model_results/hp_best.json`, y se quito el wiring de
> `build_models`/`main`. Esta seccion queda como registro de que ya se
> intento. Ver el resumen final arriba.

### Cambio
Se agrego tuning opcional por modelo via `tools/tune_hparams.py` (stdlib
`GridSearchCV`, sin Optuna). La busqueda corre sobre un subsample
estratificado con su propio CV interno (no toca los folds finales de
`main`). Metrico objetivo: PR-AUC (`average_precision`), que penaliza la
precision baja del target desbalanceado.

- Grilla enfocada por modelo: eje principal = multiplicador de
  `class_weight` para la clase positiva (`{0:1, 1:w}` con w ∈ {3,5,7} y
  `"balanced"`), mas 1–2 ejes de complejidad/regularizacion
  (`max_depth`, `learning_rate`, `min_child_weight`, `C`).
- Escribe `metadata/model_results/hp_best.json` keyed por feature_set.
- `main()` lee ese JSON y hace `set_params(**params)` por modelo cuando
  existe; si no existe, usa los defaults del pipeline. No edita defaults:
  solo override cuando el JSON esta.

Detalle de implementacion: `GridSearchCV(n_jobs=1)` afuera, estimadores
internos con `n_jobs=-1`. Anidar `n_jobs=-1` en ambos niveles causaba
deadlock por oversubscription de joblib en esta maquina (CPU ~2%, frozen).
Salida line-buffered para ver progreso real.

### Resultados (ADDEPEV3, rf_top30, 50k subsample, 3-fold, PR-AUC)
Mejores hiperparametros por modelo (PR-AUC de validacion cruzada):

| modelo | PR-AUC | class_weight (1:w) | otros |
|---|---|---|---|
| RandomForest | 0.628 | balanced | max_depth=12, max_features=sqrt |
| XGBoost | 0.631 | 1:3 | lr=0.05, max_depth=6, min_child_weight=5 |
| LightGBM | 0.634 | 1:5 | lr=0.05, max_depth=8 |
| LogisticRegression | 0.615 | 1:3 | C=10.0 |
| LinearSVC | 0.615 | 1:3 | C=1.0 |

### Conclusion
Ganancia marginal sobre el threshold tuning ya presente. ROC-AUC/PR-AUC
identicos (el multiplicador de class_weight desplaza el umbral de decision,
no el ranking), y f1/recall @ umbral casi iguales al baseline
(ej. rf_top30 10k/3-fold: f1 default 0.546/0.488/0.573/0.580/0.514 vs
tuned 0.555/0.489/0.578/0.580/0.518). La grilla confirmo que los defaults
del pipeline ya estaban cerca del optimo. Se mantiene el mecanismo (off
por defecto, activo solo si `hp_best.json` existe) pero no cambia los
numeros reportados arriba. Skipped: grids mas grandes / Optuna — add solo
si se quiere exprimir el ultimo 1% de PR-AUC.

