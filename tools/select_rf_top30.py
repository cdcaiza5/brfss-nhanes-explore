# -*- coding: utf-8 -*-
"""Selecciona las 30 mejores variables para ADDEPEV3 sobre el UNIVERSO
completo de columnas (301, menos ID/geo/peso y el target).

Dos rankers independientes sobre el mismo diseno:
  - A) RandomForest (importancia de Gini)
  - B) mutual_info_classif
Ambos se validan/comparan con permutation_importance (ROC-AUC) sobre un
fold de hold-out. El set canonico es el top-30 de RF; MI y permutation
sirven de contraste.

El diseno replica al pipeline: _RACE se mean-encodea; numericas+ordinales
se imputan por mediana y escalan; categoricas se imputan por moda y
one-hot (drop="if_binary"). No es parte del pipeline: script de
re-audit, se corre bajo demanda. Imprime ambos rankings y la comparacion
de permutation importance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import brfss_dataset_model as m  # noqa: E402

SEED = 42
SUBSAMPLE = 100_000
TOP_N = 30
N_SPLITS_HOLDOUT = 0.3  # fraccion para el fold de permutation importance

# Columnas fuera del universo de candidatos: ID, geo y pesos de diseno.
EXCLUDE = ["ADDEPEV3", "_STATE", "SEQNO", "_PSU", "_STRWT", "_WT2RAKE"]


def all_candidate_columns() -> list[str]:
    """Lee el codebook y devuelve todas las columnas menos EXCLUDE."""
    meta = pd.read_csv(m.REPO_DIR / "metadata" / "columns" / "brfss_2024_columns.csv")
    cols = [c for c in meta["name"].tolist() if c not in EXCLUDE]
    return cols


def mean_encode(train_s, y_train, test_s, alpha=10.0):
    gm = y_train.mean()
    g = pd.DataFrame({"c": train_s, "y": y_train}).groupby("c")["y"]
    means, counts = g.mean(), g.count()
    sm = (means * counts + gm * alpha) / (counts + alpha)
    return (
        train_s.map(sm).fillna(gm).astype(float),
        test_s.map(sm).fillna(gm).astype(float),
    )


def main():
    candidates = all_candidate_columns()
    # _RACE se trata aparte (mean-encoded); el resto son numericas/ordinales
    # o categoricas nominales. Para el universo completo no conocemos el tipo
    # de cada columna, asi que tratamos todo lo no-_RACE como categorica
    # nominal one-hot (BRFSS es casi todo ordinal/categorico de codigo). Las
    # pocas continuas (_BMI5, _AGE80, etc.) sobreviven bien al one-hot.
    cat_cols = [c for c in candidates if c != "_RACE"]

    df = m.load_and_clean(m.DEFAULT_XPT, candidates, "ADDEPEV3")
    y = df["target"].astype(int)

    df, df_test = train_test_split(
        df, train_size=SUBSAMPLE, stratify=y, random_state=SEED
    )
    y_tr, y_te = y.loc[df.index], y.loc[df_test.index]

    re_train, re_test = mean_encode(df["_RACE"], y_tr, df_test["_RACE"])
    X = df[cat_cols].copy()
    X_test = df_test[cat_cols].copy()
    X["_RACE_enc"] = re_train.values
    X_test["_RACE_enc"] = re_test.values
    # Todo el universo es codigo de categoria (incl. las pocas continuas como
    # _BMI5). One-hot del universo completo explota la memoria (100k x 100k+),
    # asi que label-encodeamos todas las columnas salvo _RACE_enc (que es
    # continua). RF y MI no requieren one-hot; los codigos conservan orden
    # aproximado. OrdinalEncoder maneja missing como categoria propia.
    num_cols = ["_RACE_enc"]
    # OrdinalEncoder descarta columnas sin valores observados (ej. secciones
    # no preguntadas en 2024). Las quitamos tambien de la lista de nombres
    # para que el nro de features coincida.
    all_cat = [c for c in cat_cols if df[c].notna().any()]

    enc = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             num_cols),
            ("cat", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1)),
             all_cat),
        ],
        remainder="drop",
    )
    Xf = enc.fit_transform(X)
    Xf_test = enc.transform(X_test)

    names = list(num_cols) + list(all_cat)
    feat_idx = {n: i for i, n in enumerate(names)}

    # --- Ranker A: RandomForest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, n_jobs=-1,
        class_weight="balanced", random_state=SEED,
    )
    rf.fit(Xf, y_tr)
    from sklearn.metrics import roc_auc_score
    print("RF  train AUC", round(roc_auc_score(y_tr, rf.predict_proba(Xf)[:, 1]), 4))
    print("RF   test AUC", round(roc_auc_score(y_te, rf.predict_proba(Xf_test)[:, 1]), 4))
    rf_imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    rf_top = rf_imp.head(TOP_N)

    # --- Ranker B: mutual information ---
    # Todas las columnas son codigos de categoria (discretas). MI con el
    # estimador k-NN es carisimo en 293 cols; se calcula sobre 5k filas
    # (el ranking es estable a ese tamano) y es instantaneo.
    mi_idx = np.random.RandomState(SEED).choice(len(y_tr), 5_000, replace=False)
    mi = mutual_info_classif(Xf[mi_idx], y_tr.iloc[mi_idx],
                             discrete_features=True, random_state=SEED)
    mi_imp = pd.Series(mi, index=names).sort_values(ascending=False)
    mi_top = mi_imp.head(TOP_N)

    # --- Permutation importance sobre el set canonico (top-30 RF) ---
    # Se re-entrena un RF sobre solo esas 30 columnas y se valida con
    # permutation importance en el hold-out (así la comparación es coherente:
    # el modelo de 30 cols vs. su propio input).
    perm_idx = [feat_idx[nm] for nm in rf_top.index]
    Xf_tr_top = Xf[:, perm_idx]
    Xf_test_top = Xf_test[:, perm_idx]
    rf30 = RandomForestClassifier(
        n_estimators=300, max_depth=12, n_jobs=-1,
        class_weight="balanced", random_state=SEED,
    )
    rf30.fit(Xf_tr_top, y_tr)
    print("RF30 test AUC",
          round(roc_auc_score(y_te, rf30.predict_proba(Xf_test_top)[:, 1]), 4))
    perm = permutation_import_score(rf30, Xf_test_top, y_te,
                                    list(rf_top.index))

    print(f"\n=== TOP {TOP_N} — RandomForest ===")
    for i, (nm, v) in enumerate(rf_top.items(), 1):
        print(f"{i:3} {nm:16} {v:.5f}")
    print(f"\n=== TOP {TOP_N} — Mutual Information ===")
    for i, (nm, v) in enumerate(mi_top.items(), 1):
        print(f"{i:3} {nm:16} {v:.5f}")
    print("\n=== Permutation importance (RF, ROC-AUC drop) — top 30 RF ===")
    for nm in rf_top.index:
        print(f"    {nm:16} {perm.get(nm, float('nan')):.5f}")
    print("\n=== Overlap RF ∩ MI (top 30) ===")
    overlap = set(rf_top.index) & set(mi_top.index)
    print(f"{len(overlap)} / {TOP_N} compartidas")

    # Set canonico: top-30 de RF (nombres de columna originales).
    canonical = list(rf_top.index)
    print("\n=== SET CANONICO (top-30 RF, columnas originales) ===")
    print(canonical)


def permutation_import_score(est, X_test, y_test, names, n_repeats=3):
    pi = permutation_importance(
        est, X_test, y_test, scoring="roc_auc",
        n_repeats=n_repeats, random_state=SEED, n_jobs=-1,
    )
    return dict(zip(names, pi.importances_mean))


if __name__ == "__main__":
    main()
