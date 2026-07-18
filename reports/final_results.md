# Reporte final — BRFSS 2024, ADDEPEV3

Prediccion de diagnostico de depresion de por vida (`ADDEPEV3`) sobre el
dataset completo de BRFSS 2024 (455,006 filas, 5-fold CV estratificado,
`--balancing class_weight`, umbral F1-optimo OOF, preset `default`, 5
modelos base).

## Setup

- **Target:** `ADDEPEV3` (diagnostico de depresion de por vida, 1=Si / 2=No).
  Base rate 21% (clase positiva minoritaria).
- **Datos:** BRFSS 2024, 455,006 filas, 5-fold CV estratificado (seed 42).
- **Feature set:** `default` (unico) — 30 predictores seleccionados del
  universo completo de 301 columnas (menos ID/geo/peso y el target) con
  `tools/select_rf_top30.py` (RF importance + MI + permutation como
  contraste). `_RACE` se mean-encodea por fold aparte (no cuenta en las 30).
- **Balanceo:** `class_weight="balanced"` (repondera losses sin resamplear).
- **Umbral:** F1-optimo por modelo, elegido sobre predicciones out-of-fold
  (sin leakage de test). `--threshold-metric f1`.
- **Modelos:** 5 base (RandomForest, LinearSVC, LogisticRegression, XGBoost,
  LightGBM). Sin Ensemble (probado y descartado, ver `changes_and_results.md`).
- **Salidas:** `metadata/model_results/addepev3_full_5fold/`.

## Metricas de probabilidad (independientes del umbral)

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| RandomForest      | 0.8312 | 0.6118 | 0.1709 |
| LinearSVC         | 0.8253 | 0.6071 | 0.1213 |
| LogisticRegression| 0.8260 | 0.6079 | 0.1636 |
| **XGBoost**       | **0.8433** | **0.6354** | **0.1156** |
| LightGBM          | 0.8431 | 0.6354 | 0.1568 |

## Metricas a umbral 0.5 (operacion por defecto del modelo, sin tuning)

| model | accuracy | F1 | recall | precision | specificity |
|---|---|---|---|---|---|
| RandomForest      | 0.7687 | 0.5710 | 0.7285 | 0.4695 | 0.7795 |
| LinearSVC         | 0.8333 | 0.4893 | 0.3781 | 0.6935 | 0.9552 |
| LogisticRegression| 0.7828 | 0.5733 | 0.6905 | 0.4901 | 0.8075 |
| XGBoost           | 0.8403 | 0.5383 | 0.4408 | 0.6913 | 0.9473 |
| LightGBM          | 0.7752 | 0.5849 | 0.7497 | 0.4795 | 0.7820 |

## Metricas a umbral F1-optimo (OOF-tuned, sin leakage)

| model | F1@thr | recall@thr | precision@thr | specificity@thr | threshold |
|---|---|---|---|---|---|
| RandomForest      | 0.5812 | 0.6463 | 0.5279 | 0.8452 | 0.553 |
| LinearSVC         | 0.5768 | 0.6381 | 0.5263 | 0.8462 | 0.248 |
| LogisticRegression| 0.5778 | 0.6457 | 0.5228 | 0.8421 | 0.545 |
| XGBoost           | 0.5972 | 0.6445 | 0.5564 | 0.8624 | 0.304 |
| **LightGBM**      | **0.5974** | 0.6517 | 0.5515 | 0.8581 | 0.614 |

## Lectura

- **XGBoost y LightGBM empatan en el techo de ranking** (ROC-AUC ~0.843,
  PR-AUC 0.6354). XGBoost ademas tiene el **mejor Brier (0.1156)** —
  probabilidades mejor calibradas. LightGBM tiene recall ligeramente mayor
  al umbral optimo (0.6517 vs 0.6445).
- **Threshold tuning levanta las metricas peores**: LinearSVC pasa de
  recall 0.378 (umbral 0.5) a 0.638 (umbral 0.248); XGBoost de 0.441 a
  0.645. Costo: precision baja de ~0.69 a ~0.53-0.56 y specificity de
  ~0.95 a ~0.85-0.86 en esos dos. Es el tradeoff esperado al mover el
  umbral; ROC-AUC/PR-AUC/Brier no cambian (no hay leakage).
- **Estabilidad**: std entre los 5 folds es ~0.000-0.007 — los resultados
  son muy estables en full data.
- **Techo ~0.843 ROC-AUC**: con 30 variables auto-reportadas y base rate
  21%, este es el limite real. Los experimentos descartados (HP tuning,
  interacciones degree-2, Ensemble) confirman que no hay headroom barato
  (ver `changes_and_results.md`).

## Configuracion recomendada

- **Feature set:** `default` (unico).
- **Balanceo:** `--balancing class_weight`.
- **Umbral:** `--threshold-metric f1` (default) para screening; considerar
  `--threshold-metric cost --cost-ratio 3` si se quiere priorizar recall
  todavia mas.
- **Modelo:**
  - **XGBoost** si se quiere mejor calibracion (Brier 0.116) + el techo de
    ROC-AUC/PR-AUC.
  - **LightGBM** si se quiere el mismo techo con recall ligeramente mayor
    al umbral optimo.
  - **LinearSVC** si se quiere maxima specificity a costa de recall
    (operando en umbral 0.5, sin tuning).

## Archivos de soporte

- `metadata/model_results/addepev3_full_5fold/` — `folds_*.csv` (una fila
  por modelo/fold/metrica), `summary_*.csv` (mean y std por
  modelo/metrica), 5 `cm_*.png` (matrices de confusion acumuladas, una
  por modelo al umbral F1-optimo).
- `reports/changes_and_results.md` — registro completo de cambios del
  pipeline: que se probo, que se mantuvo, que se descarto (HP tuning,
  interacciones degree-2, Ensemble) y por que, con tablas antes/despues.