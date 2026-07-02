# -*- coding: utf-8 -*-
"""BRFSS 2024 — Libreria de prediccion de salud mental.

Modulo principal del pipeline. Exporta la logica reusable y un punto de
entrada CLI (`main()`). Los wrappers brfss_dataset_model_ADDEPEV3.py y
brfss_dataset_model__MENT14D.py son puntos de entrada finos que importan
`main` desde aca y hardcodean target/feature-set por CLI.

Targets soportados:
  - ADDEPEV3: diagnostico de depresion de por vida (1=Si / 2=No).
  - _MENT14D: 14 o mas dias de mala salud mental en los ultimos 30.

Pipeline por fold:
  1. Imputacion (mediana para numericas/ordinales, moda para categoricas).
  2. Resampleo o reponderado segun `--balancing`: class_weight en cada
     modelo (`class_weight`, default), SMOTENC sintetico (`smote`),
     submuestreo aleatorio (`undersampling-random`) o NearMiss-1
     (`undersampling-nearmiss`).
  3. Encoding (StandardScaler para numericas/ordinales, OneHotEncoder para categoricas).
  4. Modelo (RandomForest, LinearSVC con calibrado, LogisticRegression, XGBoost, LightGBM).

Evaluacion: stratified k-fold cross-validation. Reporta accuracy, precision,
recall, F1, specificity, ROC-AUC, PR-AUC, Brier. Escribe un CSV por fold y
una matriz de confusion acumulada por modelo.

Uso directo:
    python brfss_dataset_model.py --target _MENT14D --feature-set recommended
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import NearMiss, RandomUnderSampler


SEED = 42
# Raiz del repo: este script vive en la raiz. Todos los paths se derivan de aca.
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_XPT = REPO_DIR / "data" / "LLCP2024.XPT_"
DEFAULT_OUT = REPO_DIR / "metadata" / "model_results"
DEFAULT_TARGET = "ADDEPEV3"

# Missing-value codes per BRFSS 2024 Codebook
# https://www.cdc.gov/brfss/annual_data/annual_2024.htm
MISSING_VALUE_MAP: dict[str, set[int]] = {
    "ADDEPEV3": {7, 9},
    "_MENT14D": {9},
    "_AGEG5YR": {14},
    "_SEX": {7, 9},
    "MARITAL": {9},
    "EDUCA": {9},
    "SMOKE100": {7, 9},
    "DRNKANY6": {7, 9},
    "EMPLOY1": {9},
    "_INCOMG1": {9},
    "GENHLTH": {7, 9},
    "_HLTHPL2": {9},
    "LSATISFY": {7, 9, 77, 99},
    "SDLONELY": {7, 9, 77, 99},
    "DIABETE4": {7, 9},
    "CVDINFR4": {7, 9},
    "ASTHMA3": {7, 9},
    "HAVARTH4": {7, 9},
    "CHCCOPD3": {7, 9},
    "CVDCRHD4": {7, 9},
    "CVDSTRK3": {7, 9},
    "PHYSHLTH": {77, 88, 99},
    "DECIDE": {7, 9},
    "DIFFWALK": {7, 9},
    "DIFFDRES": {7, 9},
    "DIFFALON": {7, 9},
    "DEAF": {7, 9},
    "BLIND": {7, 9},
    "EMTSUPRT": {7, 9, 77, 99},
    "_RACE": {9},
    "ACEDEPRS": {7, 9}, "ACEDRINK": {7, 9}, "ACEDRUGS": {7, 9},
    "ACEPRISN": {7, 9}, "ACEDIVRC": {7, 9}, "ACEPUNCH": {7, 9},
    "ACEHURT1": {7, 9}, "ACESWEAR": {7, 9}, "ACETOUCH": {7, 9},
    "ACETTHEM": {7, 9}, "ACEHVSEX": {7, 9},
    "ACEADSAF": {7, 9}, "ACEADNED": {7, 9},
    "MENTHLTH": {77, 99},
}
_BMI_MISSING = 9999  # _BMI5 is BMI*100 in the XPT

# Variables ACE con escala binaria (1=Si, 2=No). Para estos, "expuesto" = valor == 1.
ACE_BINARY = [
    "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
]
# Variables ACE con escala de frecuencia (1=Nunca, 2=Una vez, 3=Mas de una vez).
# Para estos, "expuesto" = valor >= 2.
ACE_FREQ = [
    "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM", "ACEHVSEX",
]
ACE_VARIABLES = ACE_BINARY + ACE_FREQ

CORE_NUMERIC = ["_BMI5"]
CORE_CATEGORICAL = ["_AGEG5YR", "_SEX", "MARITAL", "EDUCA", "SMOKE100", "DRNKANY6"]


# Each target has a "column" to read, and a "binarize" function that takes the
# raw column Series (after codebook NaN substitution) and returns 0/1.
TARGETS: dict[str, dict] = {
    "ADDEPEV3": {
        "column": "ADDEPEV3",
        "binarize": lambda s: (s == 1).astype(float).where(s.notna(), np.nan),
        "description": "Lifetime depression diagnosis (1=Yes / 2=No)",
    },
    "_MENT14D": {
        "column": "_MENT14D",
        "binarize": lambda s: (s == 1).astype(float).where(s.notna(), np.nan),
        "description": "14+ days of poor mental health in past 30 (computed)",
    },
}


FEATURE_SETS: dict[str, dict] = {
    "core": {
        "numeric": list(CORE_NUMERIC),
        "ordinal": ["_AGEG5YR", "EDUCA"],
        "categorical": ["_SEX", "MARITAL", "SMOKE100", "DRNKANY6"],
        "use_ace": False,
    },
    "core+ses": {
        "numeric": list(CORE_NUMERIC),
        "ordinal": ["_AGEG5YR", "EDUCA", "GENHLTH"],
        "categorical": [
            "_SEX", "MARITAL", "SMOKE100", "DRNKANY6",
            "_INCOMG1", "EMPLOY1", "_HLTHPL2",
        ],
        "use_ace": False,
    },
    "core+ses+aces": {
        "numeric": list(CORE_NUMERIC) + ["ace_score"],
        "ordinal": ["_AGEG5YR", "EDUCA", "GENHLTH", "LSATISFY", "SDLONELY"],
        "categorical": [
            "_SEX", "MARITAL", "SMOKE100", "DRNKANY6",
            "_INCOMG1", "EMPLOY1", "_HLTHPL2",
            "DIABETE4", "CVDINFR4", "ASTHMA3", "HAVARTH4",
        ],
        "use_ace": True,
    },
    "recommended": {
        "numeric": list(CORE_NUMERIC) + ["PHYSHLTH", "ace_score"],
        "ordinal": [
            "_AGEG5YR", "EDUCA", "GENHLTH",
            "LSATISFY", "SDLONELY", "EMTSUPRT",
        ],
        "categorical": [
            # Demograficos y sustancias
            "_SEX", "MARITAL", "SMOKE100", "DRNKANY6",
            # SES
            "_INCOMG1", "EMPLOY1", "_HLTHPL2",
            # Discapacidad cognitiva/fisica
            "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "DEAF", "BLIND",
            # Condiciones cronicas
            "CHCCOPD3", "CVDINFR4", "CVDCRHD4", "CVDSTRK3",
            "ASTHMA3", "HAVARTH4", "DIABETE4",
        ],
        "mean_encoded": [],
        "use_ace": True,
    },
}


@dataclass(frozen=True)
class RunConfig:
    """Configuracion inmutable de una corrida.

    `spec` ya incluye el efecto de los flags runtime (ej. `--include-race`
    agrega `_RACE` a `mean_encoded`). `columns_needed` y `feature_cols` se
    derivan una sola vez en `resolve_config` y se reutilizan en I/O y en el
    loop de CV. `balancing` es el valor codificado de `--balancing`
    (ej. `undersampling-nearmiss`).
    """
    target_name: str
    feature_set_name: str
    target_spec: dict
    spec: dict
    xpt_path: Path
    out_dir: Path
    cv: int
    seed: int
    subsample: int | None
    columns_needed: tuple[str, ...]
    feature_cols: tuple[str, ...]
    balancing: str


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """Resuelve la configuracion completa de una corrida a partir de argparse.

    Unico lugar que conoce el borde entre spec (definicion) y runtime config
    (flags como `--include-race`). Calcula `columns_needed` y `feature_cols`
    una sola vez, deduplicando.
    """
    target_spec = TARGETS[args.target]
    spec = FEATURE_SETS[args.feature_set]
    if args.include_race:
        spec = {
            **spec,
            "mean_encoded": list(spec.get("mean_encoded", [])) + ["_RACE"],
        }
    use_ace = spec["use_ace"]

    columns_needed = (
        spec["numeric"]
        + spec.get("ordinal", [])
        + spec["categorical"]
        + spec.get("mean_encoded", [])
    )
    if use_ace:
        columns_needed = list(columns_needed) + list(ACE_VARIABLES)
    columns_needed = tuple(dict.fromkeys(columns_needed))

    feature_cols = (
        spec["numeric"]
        + spec.get("ordinal", [])
        + spec["categorical"]
        + spec.get("mean_encoded", [])
    )

    return RunConfig(
        target_name=args.target,
        feature_set_name=args.feature_set,
        target_spec=target_spec,
        spec=spec,
        xpt_path=args.xpt,
        out_dir=args.out,
        cv=args.cv,
        seed=args.seed,
        subsample=args.subsample,
        columns_needed=columns_needed,
        feature_cols=tuple(feature_cols),
        balancing=args.balancing,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea argumentos de linea de comandos.

    Args:
        argv: lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con los argumentos parseados.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xpt", type=Path, default=DEFAULT_XPT,
                   help="Ruta al archivo XPT de BRFSS.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Directorio de salida para CSVs y figuras.")
    p.add_argument(
        "--target",
        choices=list(TARGETS),
        default=DEFAULT_TARGET,
        help="Variable objetivo a predecir.",
    )
    p.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS),
        default="recommended",
        help="Conjunto de predictores a usar.",
    )
    p.add_argument("--cv", type=int, default=5,
                   help="Cantidad de folds para StratifiedKFold.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Semilla aleatoria.")
    p.add_argument(
        "--include-race",
        action="store_true",
        help="Incluye _RACE como feature mean-encoded (calculado por fold). Sin este flag, _RACE no se incluye en el dataset.",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Tamano de subsample estratificado para iteracion rapida. Por defecto: dataset completo.",
    )
    p.add_argument(
        "--balancing",
        choices=[
            "class_weight",
            "smote",
            "undersampling-random",
            "undersampling-nearmiss",
        ],
        default="class_weight",
        help="Estrategia de balanceo: 'class_weight' (default, repondera "
             "losses sin resamplear), 'smote' (SMOTENC sintetico), "
             "'undersampling-random' (submuestreo aleatorio), "
             "'undersampling-nearmiss' (NearMiss-1, submuestreo por distancia).",
    )
    return p.parse_args(argv)


def load_and_clean(xpt_path: Path, columns: list[str], target_name: str) -> pd.DataFrame:
    """Carga el XPT, aplica limpieza de codigos missing y binariza el target.

    Lee unicamente las columnas necesarias (target + predictores) con
    pyreadstat. Para PHYSHLTH y MENTHLTH, mapea 88 ("None" = 0 dias) a 0
    antes de aplicar MISSING_VALUE_MAP, preservando asi la senal de "sin
    dias malos" que de otro modo se imputaria a la mediana. Luego aplica
    los codigos de missing del codebook, elimina filas con target NaN y
    binariza el target.

    Args:
        xpt_path: ruta al archivo XPT.
        columns: predictores a cargar.
        target_name: clave en TARGETS (ej. "ADDEPEV3" o "_MENT14D").

    Returns:
        DataFrame limpio con columna "target" binaria (0/1).
    """
    target_spec = TARGETS[target_name]
    target_col = target_spec["column"]
    needed = list(dict.fromkeys([target_col] + columns))
    df, _ = pyreadstat.read_xport(
        str(xpt_path), usecols=needed, encoding="cp1252"
    )
    if "_BMI5" in df.columns:
        df["_BMI5"] = df["_BMI5"].replace(_BMI_MISSING, np.nan) / 100.0
    # Mapea 88 ("None" = 0 dias) a 0 antes de tratar missing codes, para
    # no perder la informacion de "0 dias de salud fisica/mental mala".
    for col in ("PHYSHLTH", "MENTHLTH"):
        if col in df.columns:
            df[col] = df[col].replace(88, 0)
    for col, codes in MISSING_VALUE_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(list(codes), np.nan)
    # Drop rows where the target is NaN, then binarize.
    df = df.dropna(subset=[target_col])
    df["target"] = target_spec["binarize"](df[target_col])
    df = df.drop(columns=[target_col])
    return df


def add_ace_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ace_score como conteo de ACEs reportados como 'Si' o 'al menos una vez'.

    Los 11 items ACE usan dos escalas distintas:
      - ACE_BINARY (5 items): 1=Si, 2=No. Se cuenta == 1.
      - ACE_FREQ (6 items): 1=Nunca, 2=Una vez, 3=Mas de una vez. Se cuenta >= 2.
    Si todos los items ACE son NaN para una fila, ace_score queda en NaN
    (se imputara con la mediana en el pipeline).

    Args:
        df: DataFrame con las columnas ACE presentes.

    Returns:
        DataFrame con nueva columna 'ace_score' (int, en [0, 11], o NaN).
    """
    df = df.copy()
    binary_present = [v for v in ACE_BINARY if v in df.columns]
    freq_present = [v for v in ACE_FREQ if v in df.columns]
    if not binary_present and not freq_present:
        return df

    score = pd.Series(0.0, index=df.index, dtype=float)
    n_observed = pd.Series(0, index=df.index, dtype=int)

    for v in binary_present:
        s = df[v]
        # 1=Si cuenta como 1; NaN cuenta como 0 (no observado, sin aporte).
        score = score + (s == 1).astype(float).fillna(0)
        n_observed = n_observed + s.notna().astype(int)

    for v in freq_present:
        s = df[v]
        # 2 o 3 cuentan como 1; NaN cuenta como 0.
        score = score + (s >= 2).astype(float).fillna(0)
        n_observed = n_observed + s.notna().astype(int)

    # Si todos los items ACE fueron NaN, dejar ace_score en NaN para que
    # el pipeline lo impute (en vez de quedar en 0, que diria "ningun ACE").
    score[n_observed == 0] = np.nan
    df["ace_score"] = score
    return df


def smoothed_mean_encode(
    train_series: pd.Series,
    y_train: pd.Series,
    test_series: pd.Series,
    smoothing: float = 10.0,
) -> tuple[pd.Series, pd.Series]:
    """Mean encoding suavizado por categoria, ajustado solo sobre datos de entrenamiento.

    enc(c) = (n_c * mean_c + alpha * global_mean) / (n_c + alpha)

    Cada categoria se reemplaza por la media del target en entrenamiento,
    suavizada hacia la media global con peso alpha. Reduce K categorias
    nominales a 1 columna continua sin fuga de informacion (fit por fold).

    Args:
        train_series: valores categoricos de las filas de entrenamiento.
        y_train: target binario para entrenamiento.
        test_series: valores categoricos de las filas de test.
        smoothing: peso del prior (alpha). Default 10.

    Returns:
        Tupla (train_encoded, test_encoded), Series float.
    """
    global_mean = y_train.mean()
    grouped = pd.DataFrame({"c": train_series, "y": y_train}).groupby("c")["y"]
    means = grouped.mean()
    counts = grouped.count()
    smoothed = (means * counts + global_mean * smoothing) / (counts + smoothing)
    train_enc = train_series.map(smoothed).fillna(global_mean).astype(float)
    test_enc = test_series.map(smoothed).fillna(global_mean).astype(float)
    return train_enc, test_enc


def make_imputer(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Crea un ColumnTransformer que imputa numericas con mediana y categoricas con moda.

    Args:
        numeric: nombres de columnas numericas, ordinales, y mean-encoded.
        categorical: nombres de columnas categoricas nominales.

    Returns:
        ColumnTransformer listo para fit_transform.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", SimpleImputer(strategy="most_frequent"), categorical),
        ],
        remainder="drop",
    )


def make_encoder(n_num: int, n_cat: int) -> ColumnTransformer:
    """Crea un ColumnTransformer que escala numericas/ordinales y one-hot-encodea categoricas.

    Args:
        n_num: cantidad de columnas numericas/ordinales (indices 0..n_num-1).
        n_cat: cantidad de columnas categoricas (indices n_num..n_num+n_cat-1).

    Returns:
        ColumnTransformer listo para fit_transform.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(range(n_num))),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
                list(range(n_num, n_num + n_cat)),
            ),
        ],
        remainder="drop",
    )


def make_resampler(
    balancing: str, n_num: int, n_cat: int, seed: int
) -> SMOTENC | RandomUnderSampler | NearMiss | None:
    """Factory para resamplers segun el modo de balanceo.

    Args:
        balancing: valor codificado de `--balancing`
            (`smote` | `class_weight` | `undersampling-random` |
            `undersampling-nearmiss`).
        n_num: cantidad de columnas numericas/ordinales/mean-encoded.
        n_cat: cantidad de columnas categoricas nominales.
        seed: semilla aleatoria (ignorada por NearMiss, que es determinista).

    Returns:
        Instancia de resampler, o None para `class_weight` (no resamplea).
    """
    if balancing == "smote":
        return SMOTENC(
            categorical_features=list(range(n_num, n_num + n_cat)),
            random_state=seed,
        )
    if balancing == "class_weight":
        return None
    if balancing == "undersampling-random":
        return RandomUnderSampler(random_state=seed)
    if balancing == "undersampling-nearmiss":
        return NearMiss(version=1)
    raise ValueError(f"Unknown balancing method: {balancing}")


def build_models(
    seed: int, class_weight: str | dict | None = None
) -> dict[str, BaseEstimator]:
    """Construye el dict de modelos a evaluar.

    Cinco modelos:
      - RandomForest (sin calibrar, probs razonablemente calibradas para ROC-AUC).
      - LinearSVC envuelto en CalibratedClassifierCV (cv=3, sigmoid) para
        obtener predict_proba y poder calcular Brier/ROC-AUC.
      - LogisticRegression (nativamente probabilistico, ya calibrado).
      - XGBoost y LightGBM (gradient boosting; XGBoost acepta class_weight
        nativamente y lo mapea a scale_pos_weight internamente).

    Args:
        seed: semilla para todos los estimadores.
        class_weight: None | 'balanced' | dict mapeando clase -> peso.
            Se pasa a todos los modelos que lo soportan (incluido XGBoost).

    Returns:
        Dict {nombre_modelo: estimador sin fit}.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=seed,
            class_weight=class_weight,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(
                dual="auto", max_iter=2000, random_state=seed,
                class_weight=class_weight,
            ),
            cv=3,
            method="sigmoid",
        ),
        "LogisticRegression": LogisticRegression(
            solver="newton-cholesky",
            max_iter=2000,
            random_state=seed,
            class_weight=class_weight,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            class_weight=class_weight,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
            class_weight=class_weight,
        ),
    }


def main(
    defaults: list[str] | None = None,
    user_argv: list[str] | None = None,
) -> int:
    """Punto de entrada. Carga datos, hace CV, escribe resultados y figuras.

    Args:
        defaults: argumentos de linea de comandos aplicados como defaults
            (los wrappers los usan para fijar `--target` y `--feature-set`).
        user_argv: argumentos del usuario (None usa sys.argv[1:]).

    Returns:
        Codigo de salida (0 = exito).
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    argv = (defaults or []) + (user_argv if user_argv is not None else sys.argv[1:])
    args = parse_args(argv)
    config = resolve_config(args)
    np.random.seed(config.seed)
    spec = config.spec
    use_ace = spec["use_ace"]

    print(
        f"[load] reading {config.xpt_path}  usecols={len(config.columns_needed)+1}  "
        f"target={config.target_name} ({config.target_spec['description']})"
    )
    t0 = time.time()
    df = load_and_clean(
        config.xpt_path, list(config.columns_needed), config.target_name
    )
    if use_ace:
        df = add_ace_score(df)
    print(
        f"[clean] rows={len(df):,}  positive_rate={df['target'].mean():.4f}  "
        f"({time.time() - t0:.1f}s)"
    )

    if config.subsample and config.subsample < len(df):
        df, _ = train_test_split(
            df,
            train_size=config.subsample,
            stratify=df["target"],
            random_state=config.seed,
        )
        print(f"[sub] subsampled to {len(df):,} rows")

    # Las variables ordinales y mean-encoded se tratan como numericas
    # (mantienen orden / son continuas tras encoding). Las categoricas nominales
    # se one-hot-encodean.
    X = df[list(config.feature_cols)]
    y = df["target"]

    skf = StratifiedKFold(
        n_splits=config.cv, shuffle=True, random_state=config.seed
    )
    n_num = (
        len(spec["numeric"])
        + len(spec.get("ordinal", []))
        + len(spec.get("mean_encoded", []))
    )
    n_cat = len(spec["categorical"])
    resampler = make_resampler(config.balancing, n_num, n_cat, config.seed)
    class_weight_arg = (
        "balanced" if config.balancing == "class_weight" else None
    )
    models = build_models(config.seed, class_weight=class_weight_arg)
    rows: list[dict] = []
    cm_acc: dict[str, np.ndarray] = {
        n: np.zeros((2, 2), dtype=int) for n in models
    }

    print(
        f"[run] target={config.target_name}  feature_set={config.feature_set_name}  "
        f"cv={config.cv}  n_features={len(config.feature_cols)}  n={len(df):,}"
    )
    t_total = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        t_fold = time.time()

        # Mean encoding para high-cardinality cats (ajustado solo sobre train).
        # Se aplica ANTES del imputer; las columnas quedan como continuas.
        for col in spec.get("mean_encoded", []):
            train_enc, test_enc = smoothed_mean_encode(
                X_tr[col], y_tr, X_te[col]
            )
            X_tr = X_tr.copy()
            X_te = X_te.copy()
            X_tr[col] = train_enc
            X_te[col] = test_enc

        imputer = clone(make_imputer(
            spec["numeric"]
            + spec.get("ordinal", [])
            + spec.get("mean_encoded", []),
            spec["categorical"],
        ))
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        t_resample = time.time()
        if resampler is not None:
            X_tr_res, y_tr_res = clone(resampler).fit_resample(X_tr_imp, y_tr)
            resample_secs = time.time() - t_resample
            print(
                f"  [fold {fold_idx}/{config.cv}] {config.balancing}: "
                f"{resample_secs:.1f}s "
                f"({X_tr_imp.shape[0]:,} -> {X_tr_res.shape[0]:,})"
            )
        else:
            X_tr_res, y_tr_res = X_tr_imp, y_tr

        encoder = clone(make_encoder(n_num, n_cat))
        X_tr_enc = encoder.fit_transform(X_tr_res)
        X_te_enc = encoder.transform(X_te_imp)

        for name, base_model in models.items():
            m = clone(base_model)
            m.fit(X_tr_enc, y_tr_res)
            y_pred = m.predict(X_te_enc)
            y_proba = m.predict_proba(X_te_enc)[:, 1]
            cm = confusion_matrix(y_te, y_pred)
            cm_acc[name] += cm
            tn, fp, fn, tp = cm.ravel()
            metrics = {
                "accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(
                    y_te, y_pred, zero_division=0
                ),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "roc_auc": roc_auc_score(y_te, y_proba),
                "pr_auc": average_precision_score(y_te, y_proba),
                "brier": brier_score_loss(y_te, y_proba),
            }
            for k, v in metrics.items():
                rows.append(
                    {
                        "target": config.target_name,
                        "model": name,
                        "feature_set": config.feature_set_name,
                        "fold": fold_idx,
                        "metric": k,
                        "value": v,
                        "balancing": config.balancing,
                    }
                )
            print(
                f"    {name:>20s}  acc={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"roc_auc={metrics['roc_auc']:.3f}  "
                f"brier={metrics['brier']:.3f}"
            )
        print(
            f"  [fold {fold_idx}/{config.cv}] fold total: "
            f"{time.time() - t_fold:.1f}s"
        )

    out_df = pd.DataFrame(rows)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    fname_prefix = (
        f"{config.target_name}_{config.feature_set_name}_{config.balancing}"
    )
    folds_path = config.out_dir / f"folds_{fname_prefix}.csv"
    out_df.to_csv(folds_path, index=False)

    summary = (
        out_df.groupby(["model", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(config.out_dir / f"summary_{fname_prefix}.csv", index=False)

    pivot = summary.pivot(index="metric", columns="model", values="mean")
    pivot_std = summary.pivot(index="metric", columns="model", values="std")
    print(
        f"\n=== Summary (target={config.target_name}, set={config.feature_set_name}, "
        f"{config.cv}-fold) ==="
    )
    print(pivot.round(3).to_string())
    print("\n(std)")
    print(pivot_std.round(3).to_string())
    print(f"\nTotal time: {time.time() - t_total:.1f}s")

    for name, cm in cm_acc.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_title(
            f"{name} — {config.target_name} / {config.feature_set_name} ({config.cv}-fold CM)"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            config.out_dir / f"cm_{name}_{fname_prefix}.png", dpi=110
        )
        plt.close(fig)

    print(f"\nWrote:")
    print(f"  {folds_path}")
    print(f"  {config.out_dir / f'summary_{config.feature_set_name}.csv'}")
    print(f"  {config.out_dir}/cm_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
