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
  2. SMOTENC sobre el set de entrenamiento (sobre-muestrea la clase minoritaria).
  3. Encoding (StandardScaler para numericas/ordinales, OneHotEncoder para categoricas).
  4. Modelo (RandomForest, LinearSVC con calibrado, LogisticRegression).

Evaluacion: stratified k-fold cross-validation. Reporta accuracy, precision,
recall, F1, specificity, ROC-AUC, PR-AUC, Brier. Escribe un CSV por fold y
una matriz de confusion acumulada por modelo.

Uso directo:
    python brfss_dataset_model.py --target _MENT14D --feature-set recommended
"""
from __future__ import annotations

import argparse
import time
import warnings
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
            # Raza/etnia
            "_RACE",
        ],
        "use_ace": True,
    },
}


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
        default="core+ses+aces",
        help="Conjunto de predictores a usar.",
    )
    p.add_argument("--cv", type=int, default=5,
                   help="Cantidad de folds para StratifiedKFold.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Semilla aleatoria.")
    p.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Tamano de subsample estratificado para iteracion rapida. Por defecto: dataset completo.",
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


def make_imputer(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Crea un ColumnTransformer que imputa numericas con mediana y categoricas con moda.

    Args:
        numeric: nombres de columnas numericas/ordinales.
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


def build_models(seed: int) -> dict[str, BaseEstimator]:
    """Construye el dict de modelos a evaluar.

    Tres modelos:
      - RandomForest (sin calibrar, probs razonablemente calibradas para ROC-AUC).
      - LinearSVC envuelto en CalibratedClassifierCV (cv=3, sigmoid) para
        obtener predict_proba y poder calcular Brier/ROC-AUC.
      - LogisticRegression (nativamente probabilistico, ya calibrado).

    Args:
        seed: semilla para todos los estimadores.

    Returns:
        Dict {nombre_modelo: estimador sin fit}.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=seed,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(dual="auto", max_iter=2000, random_state=seed),
            cv=3,
            method="sigmoid",
        ),
        "LogisticRegression": LogisticRegression(
            solver="newton-cholesky",
            max_iter=2000,
            random_state=seed,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada. Carga datos, hace CV, escribe resultados y figuras.

    Args:
        argv: argumentos de linea de comandos (None usa sys.argv).

    Returns:
        Codigo de salida (0 = exito).
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args(argv)
    np.random.seed(args.seed)

    target_spec = TARGETS[args.target]
    spec = FEATURE_SETS[args.feature_set]
    use_ace = spec["use_ace"]

    columns_needed = (
        spec["numeric"] + spec.get("ordinal", []) + spec["categorical"]
    )
    if use_ace:
        columns_needed += ACE_VARIABLES
    columns_needed = list(dict.fromkeys(columns_needed))

    print(
        f"[load] reading {args.xpt}  usecols={len(columns_needed)+1}  "
        f"target={args.target} ({target_spec['description']})"
    )
    t0 = time.time()
    df = load_and_clean(args.xpt, columns_needed, args.target)
    if use_ace:
        df = add_ace_score(df)
    print(
        f"[clean] rows={len(df):,}  positive_rate={df['target'].mean():.4f}  "
        f"({time.time() - t0:.1f}s)"
    )

    if args.subsample and args.subsample < len(df):
        df, _ = train_test_split(
            df,
            train_size=args.subsample,
            stratify=df["target"],
            random_state=args.seed,
        )
        print(f"[sub] subsampled to {len(df):,} rows")

    # Las variables ordinales se tratan como numericas (mantienen el orden)
    # pero pasan por el mismo flujo: impute (mediana) + StandardScaler.
    feature_cols = (
        spec["numeric"] + spec.get("ordinal", []) + spec["categorical"]
    )
    X = df[feature_cols]
    y = df["target"]

    skf = StratifiedKFold(
        n_splits=args.cv, shuffle=True, random_state=args.seed
    )
    n_num = len(spec["numeric"]) + len(spec.get("ordinal", []))
    n_cat = len(spec["categorical"])
    smote = SMOTENC(
        categorical_features=list(range(n_num, n_num + n_cat)),
        random_state=args.seed,
    )

    models = build_models(args.seed)
    rows: list[dict] = []
    cm_acc: dict[str, np.ndarray] = {
        n: np.zeros((2, 2), dtype=int) for n in models
    }

    print(
        f"[run] target={args.target}  feature_set={args.feature_set}  "
        f"cv={args.cv}  n_features={len(feature_cols)}  n={len(df):,}"
    )
    t_total = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        t_fold = time.time()

        imputer = clone(make_imputer(
            spec["numeric"] + spec.get("ordinal", []),
            spec["categorical"],
        ))
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        t_sm = time.time()
        X_tr_res, y_tr_res = clone(smote).fit_resample(X_tr_imp, y_tr)
        smote_secs = time.time() - t_sm
        print(
            f"  [fold {fold_idx}/{args.cv}] SMOTENC: {smote_secs:.1f}s "
            f"({X_tr_imp.shape[0]:,} -> {X_tr_res.shape[0]:,})"
        )

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
                        "target": args.target,
                        "model": name,
                        "feature_set": args.feature_set,
                        "fold": fold_idx,
                        "metric": k,
                        "value": v,
                    }
                )
            print(
                f"    {name:>20s}  acc={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"roc_auc={metrics['roc_auc']:.3f}  "
                f"brier={metrics['brier']:.3f}"
            )
        print(
            f"  [fold {fold_idx}/{args.cv}] fold total: "
            f"{time.time() - t_fold:.1f}s"
        )

    out_df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    fname_prefix = f"{args.target}_{args.feature_set}"
    folds_path = args.out / f"folds_{fname_prefix}.csv"
    out_df.to_csv(folds_path, index=False)

    summary = (
        out_df.groupby(["model", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(args.out / f"summary_{fname_prefix}.csv", index=False)

    pivot = summary.pivot(index="metric", columns="model", values="mean")
    pivot_std = summary.pivot(index="metric", columns="model", values="std")
    print(
        f"\n=== Summary (target={args.target}, set={args.feature_set}, "
        f"{args.cv}-fold) ==="
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
            f"{name} — {args.target} / {args.feature_set} ({args.cv}-fold CM)"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            args.out / f"cm_{name}_{fname_prefix}.png", dpi=110
        )
        plt.close(fig)

    print(f"\nWrote:")
    print(f"  {folds_path}")
    print(f"  {args.out / f'summary_{args.feature_set}.csv'}")
    print(f"  {args.out}/cm_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
