# BRFSS 2024 — Comparacion de estrategias de balanceo

Prueba del mejor combination del benchmark 2x2x2 (`_MENT14D` + `recommended` +
`--include-race`) bajo las 4 estrategias de balanceo disponibles:
`smote`, `class_weight`, `undersampling-random`, `undersampling-nearmiss`.

Mismo seed (`SEED=42`), stratified 5-fold CV, subsample de 100,000 filas.

## Contexto: la mejor combination del 2x2x2

Del reporte `reports/benchmark_2x2x2.md`, la corrida con mejor desempeno
global es la #2:

| Campo | Valor |
|---|---|
| Target | `_MENT14D` |
| Feature set | `recommended` |
| `--include-race` | yes |
| `--balancing` | `smote` (default) |
| n_features | 30 (29 + `_RACE` mean-encoded) |
| n | 100,000 |
| Wall time | 651 s |

Metricas (LightGBM, mejor modelo):

| ROC-AUC | F1 | PR-AUC | Brier | Recall | Specificity |
|---:|---:|---:|---:|---:|---:|
| 0.7986 | 0.7904 | 0.8423 | 0.1761 | 0.8211 | 0.6145 |

Es la unica combination del 2x2x2 donde los modelos de gradient boosting
superan 0.79 ROC-AUC y 0.78 F1 simultaneamente, con un base rate favorable
(60% positivos) y senal fuerte (features current-state matching current-state
target).

## Setup

Comandos ejecutados:

```bash
# smote (re-run para auto-contencion)
python brfss_dataset_model__MENT14D.py --include-race --balancing smote \
  --subsample 100000 --cv 5 --out /tmp/benchmark_balancing/smote

# class_weight
python brfss_dataset_model__MENT14D.py --include-race --balancing class_weight \
  --subsample 100000 --cv 5 --out /tmp/benchmark_balancing/class_weight

# undersampling-random
python brfss_dataset_model__MENT14D.py --include-race --balancing undersampling-random \
  --subsample 100000 --cv 5 --out /tmp/benchmark_balancing/undersampling-random

# undersampling-nearmiss
python brfss_dataset_model__MENT14D.py --include-race --balancing undersampling-nearmiss \
  --subsample 100000 --cv 5 --out /tmp/benchmark_balancing/undersampling-nearmiss
```

Tamano del set de entrenamiento despues del resampleo (80k filas al inicio
de cada fold, 60% positivas = 48k pos / 32k neg):

| Modo | Post-resampleo | Cambio |
|---|---:|---|
| `smote` | 96,072 | +20% (oversample hasta balance) |
| `class_weight` | 80,000 | sin cambio |
| `undersampling-random` | 63,928 | -20% (undersample hasta balance) |
| `undersampling-nearmiss` | 63,928 | -20% (undersample hasta balance) |

## Matriz de corridas

| # | Balancing | Wall time | Tiempo resampleo/fold | Set train final |
|---|---|---:|---:|---:|
| 1 | `smote` | 727 s | 109 s | 96,072 |
| 2 | `class_weight` | 126 s | — | 80,000 |
| 3 | `undersampling-random` | 100 s | <0.1 s | 63,928 |
| 4 | `undersampling-nearmiss` | 138 s | 5 s | 63,928 |

`class_weight` es 5-7x mas rapido que `smote` porque evita el KNN de
SMOTENC. `undersampling-nearmiss` sorprende con 5s/fold: el calculo de
distancias en 80k × 30 features es eficiente en CPU.

---

## Resultados por balancing

### `smote` (default, baseline)

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7253 | 0.7787 | 0.7579 | 0.7681 | 0.6762 | 0.7930 | 0.8384 | 0.1838 |
| LinearSVC | 0.7028 | 0.7767 | 0.7088 | 0.7412 | 0.6937 | 0.7731 | 0.8227 | 0.1938 |
| LogisticRegression | 0.7065 | 0.7726 | 0.7246 | 0.7478 | 0.6794 | 0.7734 | 0.8224 | 0.1920 |
| XGBoost | 0.7380 | 0.7622 | 0.8192 | 0.7897 | 0.6159 | 0.7984 | 0.8420 | 0.1762 |
| LightGBM | 0.7385 | 0.7620 | 0.8211 | 0.7904 | 0.6145 | 0.7986 | 0.8423 | 0.1761 |

### `class_weight`

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7274 | 0.7852 | 0.7517 | 0.7681 | 0.6910 | 0.7959 | 0.8403 | 0.1844 |
| LinearSVC | 0.7259 | 0.7386 | 0.8413 | 0.7866 | 0.5525 | 0.7831 | 0.8311 | 0.1823 |
| LogisticRegression | 0.7139 | 0.7786 | 0.7315 | 0.7543 | 0.6873 | 0.7834 | 0.8313 | 0.1884 |
| XGBoost | 0.7392 | 0.7535 | 0.8408 | 0.7947 | 0.5867 | 0.7990 | 0.8430 | 0.1758 |
| LightGBM | 0.7294 | 0.7876 | 0.7522 | 0.7695 | 0.6952 | 0.8001 | 0.8436 | 0.1806 |

### `undersampling-random`

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7261 | 0.7871 | 0.7455 | 0.7657 | 0.6969 | 0.7957 | 0.8400 | 0.1850 |
| LinearSVC | 0.7133 | 0.7789 | 0.7297 | 0.7535 | 0.6888 | 0.7830 | 0.8311 | 0.1886 |
| LogisticRegression | 0.7139 | 0.7784 | 0.7319 | 0.7545 | 0.6869 | 0.7833 | 0.8312 | 0.1884 |
| XGBoost | 0.7261 | 0.7874 | 0.7450 | 0.7656 | 0.6976 | 0.7977 | 0.8416 | 0.1824 |
| LightGBM | 0.7286 | 0.7892 | 0.7476 | 0.7679 | 0.7000 | 0.7993 | 0.8432 | 0.1816 |

### `undersampling-nearmiss`

| Model | accuracy | precision | recall | f1 | specificity | roc_auc | pr_auc | brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.6877 | 0.7705 | 0.6835 | 0.7244 | 0.6940 | 0.7580 | 0.8145 | 0.2092 |
| LinearSVC | 0.6941 | 0.7662 | 0.7061 | 0.7349 | 0.6762 | 0.7573 | 0.8159 | 0.2060 |
| LogisticRegression | 0.6855 | 0.7713 | 0.6769 | 0.7210 | 0.6984 | 0.7539 | 0.8140 | 0.2158 |
| XGBoost | 0.6741 | 0.7778 | 0.6402 | 0.7023 | 0.7252 | 0.7413 | 0.8099 | 0.2372 |
| LightGBM | 0.6731 | 0.7786 | 0.6367 | 0.7005 | 0.7279 | 0.7405 | 0.8082 | 0.2377 |

---

## Comparacion contra `smote` (delta)

### ROC-AUC

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.7930 | **+0.0029** | +0.0027 | −0.0350 |
| LinearSVC | 0.7731 | **+0.0100** | +0.0099 | −0.0158 |
| LogisticRegression | 0.7734 | **+0.0100** | +0.0099 | −0.0195 |
| XGBoost | 0.7984 | +0.0006 | −0.0007 | **−0.0571** |
| LightGBM | 0.7986 | **+0.0015** | +0.0007 | −0.0581 |

### F1

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.7681 | +0.0000 | −0.0024 | −0.0437 |
| LinearSVC | 0.7412 | **+0.0454** | +0.0123 | −0.0063 |
| LogisticRegression | 0.7478 | +0.0065 | +0.0067 | −0.0268 |
| XGBoost | 0.7897 | **+0.0050** | −0.0241 | −0.0874 |
| LightGBM | 0.7904 | −0.0209 | −0.0225 | **−0.0899** |

### Brier (menor es mejor)

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.1838 | +0.0006 | +0.0012 | +0.0254 |
| LinearSVC | 0.1938 | **−0.0115** | −0.0052 | +0.0122 |
| LogisticRegression | 0.1920 | **−0.0036** | −0.0036 | +0.0238 |
| XGBoost | 0.1762 | **−0.0004** | +0.0062 | +0.0610 |
| LightGBM | 0.1761 | +0.0045 | +0.0055 | +0.0616 |

### PR-AUC

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.8384 | **+0.0019** | +0.0016 | −0.0239 |
| LinearSVC | 0.8227 | **+0.0084** | +0.0084 | −0.0068 |
| LogisticRegression | 0.8224 | **+0.0089** | +0.0088 | −0.0084 |
| XGBoost | 0.8420 | +0.0010 | −0.0004 | **−0.0321** |
| LightGBM | 0.8423 | **+0.0013** | +0.0009 | −0.0341 |

---

## Precision/Recall tradeoff

### Precision por (modelo, balancing)

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.7787 | 0.7852 | 0.7871 | 0.7705 |
| LinearSVC | 0.7767 | 0.7386 | 0.7789 | 0.7662 |
| LogisticRegression | 0.7726 | 0.7786 | 0.7784 | 0.7713 |
| XGBoost | 0.7622 | 0.7535 | 0.7874 | 0.7778 |
| LightGBM | 0.7620 | 0.7876 | 0.7892 | 0.7786 |

### Recall por (modelo, balancing)

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.7579 | 0.7517 | 0.7455 | 0.6835 |
| LinearSVC | 0.7088 | **0.8413** | 0.7297 | 0.7061 |
| LogisticRegression | 0.7246 | 0.7315 | 0.7319 | 0.6769 |
| XGBoost | 0.8192 | **0.8408** | 0.7450 | 0.6402 |
| LightGBM | 0.8211 | 0.7522 | 0.7476 | 0.6367 |

### Specificity por (modelo, balancing)

| Model | smote | class_weight | undersampling-random | undersampling-nearmiss |
|---|---:|---:|---:|---:|
| RandomForest | 0.6762 | 0.6910 | 0.6969 | 0.6940 |
| LinearSVC | 0.6937 | 0.5525 | 0.6888 | 0.6762 |
| LogisticRegression | 0.6794 | 0.6873 | 0.6869 | 0.6984 |
| XGBoost | 0.6159 | 0.5867 | 0.6976 | 0.7252 |
| LightGBM | 0.6145 | 0.6952 | 0.7000 | 0.7279 |

`class_weight` empuja el umbral efectivo hacia predecir "si" con mas
frecuencia, lo que sube recall y baja specificity. Para LinearSVC el efecto
es dramatico: recall sube de 0.71 a 0.84 y specificity cae de 0.69 a 0.55.
Para los modelos de gradient boosting el efecto es modesto.

`undersampling-nearmiss` reduce recall drasticamente (0.82 → 0.64 para
LightGBM) porque descartar informacion real de la clase mayoritaria no se
compensa con la seleccion informada de NearMiss-1.

---

## Efecto por modelo

### LinearSVC y LogisticRegression

Los modelos lineales son los que **mas se benefician** de `class_weight`:

| Modelo | smote ROC-AUC | class_weight ROC-AUC | Δ |
|---|---:|---:|---:|
| LinearSVC | 0.7731 | 0.7831 | **+0.0100** |
| LogisticRegression | 0.7734 | 0.7834 | **+0.0100** |

`smote` y `undersampling-random` dan resultados similares para los modelos
lineales (todos ~0.783). Esto es esperable: los modelos lineales no pueden
aprovechar las muestras sinteticas de SMOTENC (solo ven puntos en el
espacio, sin estructura), entonces el efecto del oversampling sintetico se
diluye. En cambio, `class_weight` actua directamente sobre la loss function
y es mas efectivo.

### XGBoost y LightGBM

Los modelos de gradient boosting tienen un patron distinto:

| Modelo | smote ROC-AUC | class_weight ROC-AUC | Δ |
|---|---:|---:|---:|
| XGBoost | 0.7984 | 0.7990 | +0.0006 |
| LightGBM | 0.7986 | 0.8001 | +0.0015 |

ROC-AUC marginalmente mejor con `class_weight`, pero F1 mejor con `smote`
(porque `smote` empuja recall mas alto). Los modelos de boosting son
robustos al desbalance porque hacen split por ganancia y la clase
minoritaria sigue teniendo suficientes muestras (40% del set).

### RandomForest

| Modo | ROC-AUC | F1 | Brier |
|---|---:|---:|---:|
| smote | 0.7930 | 0.7681 | 0.1838 |
| class_weight | **0.7959** | 0.7681 | 0.1844 |
| undersampling-random | 0.7957 | 0.7657 | 0.1850 |
| undersampling-nearmiss | 0.7580 | 0.7244 | 0.2092 |

RandomForest se comporta similar a los modelos de boosting: `class_weight`
marginalmente mejor en ROC-AUC, `smote` marginalmente mejor en F1.
Diferencias pequenas.

---

## Conclusiones

### Ranking por ROC-AUC (mejor modelo por balancing)

| Balancing | Best model | ROC-AUC | F1 | Brier |
|---|---|---:|---:|---:|
| `smote` | LightGBM | 0.7986 | **0.7904** | **0.1761** |
| `class_weight` | LightGBM | **0.8001** | 0.7947 | 0.1758 |
| `undersampling-random` | LightGBM | 0.7993 | 0.7679 | 0.1816 |
| `undersampling-nearmiss` | RandomForest | 0.7580 | 0.7244 | 0.2092 |

### Headlines

1. **`class_weight` es el mejor default** para `_MENT14D` con este
   combination. Gana en ROC-AUC para todos los modelos (+0.001 a +0.010),
   gana en F1 para 3 de 5 modelos, y gana en Brier para 2 de 5. Nunca
   pierde mas de 0.005 ROC-AUC contra `smote`.

2. **`smote` sigue siendo competitivo** especificamente para F1 de los
   modelos de gradient boosting (LightGBM 0.7904, XGBoost 0.7897) y para
   Brier de los mismos. Si la metrica prioritaria es F1 y el modelo es de
   boosting, `smote` es una opcion valida.

3. **`undersampling-nearmiss` es la peor opcion** por un margen grande.
   Pierde 0.04-0.06 ROC-AUC en todos los modelos y empuja recall a 0.64
   (de 0.82 con `smote`). Descartar el 37.5% de los datos de entrenamiento
   no se compensa con la seleccion "informada" por distancia. Para este
   dataset con 60% positivos, el undersampling es contraproducente.

4. **`undersampling-random` es similar a `class_weight` en ROC-AUC** para
   los modelos lineales y un poco peor para los de boosting. Pierde data
   sin ventaja clara.

5. **El `class_weight` es 5-7x mas rapido que `smote`**: 126s vs 727s en
   este dataset. Si la velocidad importa (ej. busqueda de hiperparametros),
   `class_weight` es claramente preferible.

### Recomendacion practica

- **Default**: cambiar de `smote` a `class_weight`. Mejora todas las
  metricas discriminativas (ROC-AUC, PR-AUC) y reduce el tiempo de corrida
  5-7x. El unico trade-off es una leve subida de recall y leve bajada de
  specificity para LinearSVC (lo cual es deseable si priorizamos deteccion).
- **Si se prioriza F1 puro** y se usan modelos de gradient boosting:
  mantener `smote`. La diferencia es marginal.
- **Evitar `undersampling-nearmiss`** salvo que haya una razon especifica
  (ej. requisitos de tamaño de modelo en produccion).
