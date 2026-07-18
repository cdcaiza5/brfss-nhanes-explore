# -*- coding: utf-8 -*-
"""BRFSS 2024 — Auditoria de features y comparacion de targets.

Carga el XPT 2024, limpia segun el codebook y calcula, para cada variable:
  * Tier de conocimiento de dominio (strong / moderate / weak / leakage / skip).
  * Asociacion univariada con ADDEPEV3 (point-biserial para numericas, V de Cramer para categoricas).
  * Informacion mutua con ADDEPEV3.
  * Importancia por RandomForest frente a ADDEPEV3.

Adicionalmente evalua 4 targets candidatos:
  * ADDEPEV3 (el target actual).
  * _MENT14D (14+ dias de mala salud mental en los ultimos 30).
  * MENTHLTH_b14 (idem, derivado del MENTHLTH crudo).
  * LSATISFY_dissat (insatisfecho o muy insatisfecho).

Salidas (en metadata/audit/):
  feature_audit.csv     - una fila por variable con todos los puntajes.
  feature_audit.md      - top-30 por metrica + set recomendado.
  target_comparison.csv
  target_comparison.md

Uso:
    python tools/audit.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pyreadstat

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


# Raiz del repo: este script vive en tools/, un nivel abajo. Todos los paths
# se derivan de aca.
REPO_DIR = Path(__file__).resolve().parent.parent
XPT_PATH = REPO_DIR / "data" / "LLCP2024.XPT_"
OUT_DIR = REPO_DIR / "metadata" / "audit"
SEED = 42
SUBSAMPLE = 100_000
CURRENT_TARGET = "ADDEPEV3"

# Missing-value codes per BRFSS 2024 Codebook
MISSING_VALUE_MAP: dict[str, set[int]] = {
    CURRENT_TARGET: {7, 9},
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
    "ACEDEPRS": {7, 9}, "ACEDRINK": {7, 9}, "ACEDRUGS": {7, 9},
    "ACEPRISN": {7, 9}, "ACEDIVRC": {7, 9}, "ACEPUNCH": {7, 9},
    "ACEHURT1": {7, 9}, "ACESWEAR": {7, 9}, "ACETOUCH": {7, 9},
    "ACETTHEM": {7, 9}, "ACEHVSEX": {7, 9},
    "ACEADSAF": {7, 9}, "ACEADNED": {7, 9},
    "_MENT14D": {9},
    "_PHYS14D": {9},
    "_RFHLTH": {9},
    "POORHLTH": {77, 88, 99},
    "MENTHLTH": {77, 88, 99},
    "PHYSHLTH": {77, 88, 99},
}
_BMI_MISSING = 9999

# ----- Domain tiers -----
# Variables with a strong prior association with depression in the literature.
STRONG_PRIOR: set[str] = {
    "MENTHLTH", "LSATISFY", "SDLONELY", "EMTSUPRT",
    "CIMEMLO1", "CDWORRY", "CDDISCU1", "CDHOUS1", "CDSOCIA1",
    "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3", "ASTHNOW",
    "CHCSCNC1", "CHCOCNC1", "CHCCOPD3", "CHCKDNY2", "HAVARTH4",
    "DIABETE4", "DIABAGE4", "PREDIAB2",
    "DEAF", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON",
    "GENHLTH", "PHYSHLTH",
    "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
    "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM",
    "ACEHVSEX", "ACEADSAF", "ACEADNED",
    "CAREGIV1", "CRGVPRB4", "CRGVALZD",
}

MODERATE_PRIOR: set[str] = {
    "INCOME3", "EDUCA", "EMPLOY1", "RENTHOM1",
    "PRIMINS2", "MEDCOST1", "CHECKUP1", "PERSDOC3",
    "FOODSTMP", "SDHFOOD1", "SDHBILLS", "SDHUTILS", "SDHTRNSP", "SDHEMPLY",
    "EXERANY2", "LASTDEN4", "RMVTETH4",
    "SMOKE100", "SMOKDAY2", "USENOW3", "ECIGNOW3",
    "DRNKANY6", "ALCDAY4", "AVEDRNK4", "DRNK3GE5", "MAXDRNKS",
    "MARIJAN1", "MARJSMOK", "MARJEAT", "MARJVAPE", "MARJDAB", "MARJOTHR",
    "WEIGHT2", "HEIGHT3",
    "MARITAL", "CHILDREN", "VETERAN3",
    "HOWSAFE1", "FIREARM5",
    "FLUSHOT7", "FLSHTMY3", "PNEUVAC4", "HIVTST7",
    "SHINGLE2", "HPVADVC4", "TETANUS1",
    "CRGVREL5", "CRGVNURS", "CRGVPER2", "CRGVHOU2", "CRGVHRS2", "CRGVLNG2",
}

# Variables that would leak the target if it is mental-health-related.
LEAKAGE: set[str] = {
    "_MENT14D", "_PHYS14D", "_RFHLTH", "POORHLTH", "MENTHLTH",
}

# Variables to skip entirely: weights, IDs, file metadata, interview mode.
SKIP_NAMES: set[str] = {
    "SEQNO", "IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH", "DISPCODE",
    "QSTVER", "QSTLANG", "CTELENM1", "PVTRESD1", "COLGHOUS", "STATERE1",
    "CELPHON1", "LADULT1", "RESPSLC1", "LANDSEX3", "SAFETIME", "CTELNUM1",
    "CELLFON5", "CADULT1", "CELLSEX3", "PVTRESD3", "CCLGHOUS", "CSTATE1",
    "LANDLINE", "SEXVAR", "HPVDSHT", "ICFQSTVR", "STATERE1",
    "_PSU", "_LLCPWT", "_LLCPWT2", "_DUALUSE", "_DUALCOR",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_IMPRACE",
    "RCSGEND1", "RCSXBRTH", "RCSRLTN2", "CASTHDX2", "CASTHNO2", "CAGEG",
    "_CHISPNC", "_CRACE1", "_CLLCPWT",
}


def classify_tier(name: str) -> str:
    """Clasifica una variable en tier segun conocimiento de dominio.

    Args:
        name: nombre de la columna BRFSS.

    Returns:
        Uno de: 'skip', 'leakage', 'strong', 'moderate', 'weak'.
    """
    if name in SKIP_NAMES:
        return "skip"
    if name in LEAKAGE:
        return "leakage"
    if name in STRONG_PRIOR:
        return "strong"
    if name in MODERATE_PRIOR:
        return "moderate"
    return "weak"


def is_categorical(name: str, series: pd.Series) -> bool:
    """Decide si una variable debe tratarse como categorica nominal.

    Heuristica: aparece en MISSING_VALUE_MAP, tiene <=15 valores unicos,
    o su nombre empieza con guion bajo (variables computadas por BRFSS).

    Args:
        name: nombre de la columna.
        series: la serie de la columna (se ignora su contenido, solo nunique).

    Returns:
        True si la variable es categorica.
    """
    if name in MISSING_VALUE_MAP:
        return True
    n_unique = series.nunique(dropna=True)
    if n_unique <= 15:
        return True
    if name.startswith("_"):
        return True
    return False


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramer entre dos series categoricas (asociacion marginal).

    Args:
        x: serie categorica.
        y: serie categorica (tipicamente el target binarizado).

    Returns:
        V de Cramer en [0, 1]. Devuelve 0.0 si la tabla de contingencia es
        degenerada.
    """
    cm = pd.crosstab(x, y)
    if cm.size == 0 or min(cm.shape) < 2:
        return 0.0
    n = cm.values.sum()
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    chi2 = ((cm.values - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    r, k = cm.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def point_biserial(x: pd.Series, y: pd.Series) -> float:
    """Correlacion point-biserial entre una serie numerica y un target binario.

    Args:
        x: serie numerica.
        y: serie binaria (0/1).

    Returns:
        Coeficiente de correlacion en [-1, 1]. Devuelve 0.0 si alguna
        serie tiene desviacion estandar cero.
    """
    x_clean = x.dropna()
    y_clean = y.loc[x_clean.index]
    if x_clean.std() == 0 or y_clean.std() == 0:
        return 0.0
    return float(np.corrcoef(x_clean, y_clean)[0, 1])


def make_target(name: str, df: pd.DataFrame) -> pd.Series:
    """Devuelve un target binario (1=positivo, 0=negativo, NaN=missing).

    Acepta el target actual (CURRENT_TARGET, ya renombrado a 'target'),
    targets derivados (MENTHLTH_b14, LSATISFY_dissat) que se calculan en
    el momento, y targets directos presentes como columna en el df.

    Args:
        name: nombre del target (CURRENT_TARGET o derivado).
        df: DataFrame con las columnas necesarias.

    Returns:
        Serie binaria, mismo indice que df.
    """
    # The current target was renamed to "target" earlier in the pipeline.
    if name == CURRENT_TARGET:
        s = df["target"].astype(float)
        return s
    # Derived targets: compute on the fly.
    if name == "MENTHLTH_b14":
        if "MENTHLTH" not in df.columns:
            raise KeyError("MENTHLTH not in df")
        s = df["MENTHLTH"].copy()
        s = s.replace(list(MISSING_VALUE_MAP.get("MENTHLTH", set())), np.nan)
        return (s >= 14).astype(float).where(s.notna(), np.nan)
    if name == "LSATISFY_dissat":
        if "LSATISFY" not in df.columns:
            raise KeyError("LSATISFY not in df")
        s = df["LSATISFY"].copy()
        s = s.replace(list(MISSING_VALUE_MAP.get("LSATISFY", set())), np.nan)
        return (s >= 3).astype(float).where(s.notna(), np.nan)
    # Otherwise, the column is present directly.
    if name not in df.columns:
        raise KeyError(f"Column {name!r} not in df")
    s = df[name].copy()
    if name in MISSING_VALUE_MAP:
        s = s.replace(list(MISSING_VALUE_MAP[name]), np.nan)
    if name == "_MENT14D":
        return (s == 1).astype(float).where(s.notna(), np.nan)
    raise ValueError(f"Unknown target: {name}")


def evaluate_target(name: str, df: pd.DataFrame, feats: list[str]) -> dict[str, float | int | str]:
    """Evalua un target con un baseline LR simple (impute + LR, sin SMOTENC) en 3-fold CV.

    Args:
        name: nombre del target a evaluar.
        df: DataFrame limpio con las columnas necesarias.
        feats: lista de features a usar como predictores.

    Returns:
        Dict con keys: target, prevalence, missingness, n_valid,
        baseline_roc_auc, baseline_f1.
    """
    y = make_target(name, df)
    valid = y.notna()
    sub = df.loc[valid, feats].copy()
    yv = y.loc[valid].astype(int)

    # 3-fold CV with simple impute+LR
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs, f1s = [], []
    for tr, te in skf.split(sub, yv):
        imputer = SimpleImputer(strategy="most_frequent")
        X_tr = imputer.fit_transform(sub.iloc[tr])
        X_te = imputer.transform(sub.iloc[te])
        clf = LogisticRegression(
            solver="newton-cholesky", max_iter=1000, random_state=SEED)
        clf.fit(X_tr, yv.iloc[tr])
        proba = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(yv.iloc[te], proba))
        f1s.append(f1_score(yv.iloc[te], (proba >= 0.5).astype(int), zero_division=0))
    return {
        "target": name,
        "prevalence": float(yv.mean()),
        "missingness": float(1 - valid.mean()),
        "n_valid": int(valid.sum()),
        "baseline_roc_auc": float(np.mean(aucs)),
        "baseline_f1": float(np.mean(f1s)),
    }


def write_audit_md(audit: pd.DataFrame, path: Path) -> None:
    """Escribe el reporte markdown del audit de features.

    Incluye: tier counts, top-30 por metrica, strong-prior no usados, top-15 recomendados.

    Args:
        audit: DataFrame con columnas name, tier, rf_importance, univariate, mi.
        path: ruta destino del archivo .md.
    """
    lines = ["# BRFSS 2024 feature audit\n"]

    lines.append("## Tier counts\n")
    lines.append("| tier | n |")
    lines.append("|---|---|")
    for t in ["strong", "moderate", "weak", "leakage", "skip"]:
        n = int((audit["tier"] == t).sum())
        lines.append(f"| {t} | {n} |")
    lines.append("")

    for metric, label in [
        ("rf_importance", "RandomForest importance"),
        ("univariate", "Univariate (point-biserial | Cramér's V)"),
        ("mi", "Mutual information"),
    ]:
        lines.append(f"## Top-30 by {label}\n")
        lines.append("| rank | name | tier | score |")
        lines.append("|---|---|---|---|")
        sub = audit[audit["tier"] != "skip"].sort_values(metric, ascending=False).head(30)
        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            lines.append(f"| {i} | {row['name']} | {row['tier']} | {row[metric]:.4f} |")
        lines.append("")

    # Strong-prior variables not in current core+ses+aces
    current_set = {
        "_AGEG5YR", "_SEX", "_BMI5", "MARITAL", "EDUCA", "SMOKE100", "DRNKANY6",
        "_INCOMG1", "EMPLOY1", "GENHLTH", "_HLTHPL2",
        "LSATISFY", "SDLONELY",
        "DIABETE4", "CVDINFR4", "ASTHMA3", "HAVARTH4",
    }
    lines.append("## Strong-prior variables not yet in the model\n")
    lines.append("| name | tier | rf | univariate | mi |")
    lines.append("|---|---|---|---|---|")
    sub = audit[(audit["tier"] == "strong") & (~audit["name"].isin(current_set))]
    sub = sub.sort_values("rf_importance", ascending=False).head(40)
    for _, row in sub.iterrows():
        lines.append(
            f"| {row['name']} | {row['tier']} | {row['rf_importance']:.4f} | "
            f"{row['univariate']:.4f} | {row['mi']:.4f} |"
        )
    lines.append("")

    # Top-15 recommended
    lines.append("## Top-15 recommended for the next feature set\n")
    lines.append("Combined score: sum of (rf rank + univariate rank + mi rank) across non-skip, non-leakage variables. Lower is better.\n")
    lines.append("| rank | name | tier |")
    lines.append("|---|---|---|")
    sub = audit[audit["tier"].isin(["strong", "moderate", "weak"])].copy()
    for col in ["rf_importance", "univariate", "mi"]:
        sub[col + "_rank"] = sub[col].rank(ascending=False)
    sub["combined"] = (
        sub["rf_importance_rank"] + sub["univariate_rank"] + sub["mi_rank"]
    )
    sub = sub.sort_values("combined").head(15)
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        lines.append(f"| {i} | {row['name']} | {row['tier']} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_target_md(targets_df: pd.DataFrame, path: Path) -> None:
    """Escribe el reporte markdown de comparacion de targets.

    Args:
        targets_df: DataFrame con columnas target, prevalence, missingness, etc.
        path: ruta destino del archivo .md.
    """
    lines = ["# BRFSS 2024 target comparison\n"]
    lines.append("Same `core+ses+aces` feature set, 3-fold CV, simple LR baseline (impute + LR, no SMOTENC).\n")
    lines.append("| target | prevalence | missingness | n_valid | baseline_roc_auc | baseline_f1 |")
    lines.append("|---|---|---|---|---|---|")
    for row in targets_df.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.prevalence:.3f} | {row.missingness:.3f} | "
            f"{row.n_valid:,} | {row.baseline_roc_auc:.3f} | {row.baseline_f1:.3f} |"
        )
    lines.append("")

    # Cross-tab vs ADDEPEV3 (for the non-ADDEPEV3 targets)
    lines.append("\n## Cross-tab vs ADDEPEV3 (current target)\n")
    lines.append(
        "Shows how much each candidate target overlaps with the current target. "
        "If overlap > 90%, switching target adds little new information.\n"
    )
    for name in ["_MENT14D", "MENTHLTH_b14", "LSATISFY_dissat"]:
        if name in targets_df["target"].values:
            lines.append(f"### {name}\n")
            lines.append("(computed inline during evaluation)\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Punto de entrada. Carga datos, calcula scores por variable, evalua targets, escribe reportes.

    Returns:
        Codigo de salida (0 = exito).
    """
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading all 301 columns from XPT ...")
    t0 = time.time()
    df, _ = pyreadstat.read_xport(str(XPT_PATH), encoding="cp1252")
    print(f"  shape={df.shape}  ({time.time()-t0:.1f}s)")

    print("[2] Cleaning per codebook ...")
    t0 = time.time()
    if "_BMI5" in df.columns:
        df["_BMI5"] = df["_BMI5"].replace(_BMI_MISSING, np.nan) / 100.0
    for col, codes in MISSING_VALUE_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(list(codes), np.nan)
    df = df.dropna(subset=[CURRENT_TARGET])
    df["target"] = (df[CURRENT_TARGET] == 1).astype(int)
    df = df.drop(columns=[CURRENT_TARGET])
    print(
        f"  cleaned to {df.shape}  positive_rate={df['target'].mean():.4f}  "
        f"({time.time()-t0:.1f}s)"
    )

    print("[3] Stratified subsample to 100k ...")
    df_sub, _ = train_test_split(
        df, train_size=SUBSAMPLE, stratify=df["target"], random_state=SEED
    )
    print(f"  subsampled to {df_sub.shape}")

    feature_cols = [c for c in df_sub.columns if c != "target"]
    X = df_sub[feature_cols]
    y = df_sub["target"]

    print("[4] Tier classification ...")
    tiers = {c: classify_tier(c) for c in feature_cols}
    tier_counts = {t: sum(1 for v in tiers.values() if v == t)
                   for t in ["strong", "moderate", "weak", "leakage", "skip"]}
    print(f"  {tier_counts}")

    usable_cols = [c for c in feature_cols if tiers[c] != "skip"]
    X_usable = X[usable_cols]
    print(f"  {len(usable_cols)} usable columns")

    print("[5] Univariate scores (point-biserial / Cramér's V) ...")
    t0 = time.time()
    univariate = {}
    for col in usable_cols:
        try:
            if is_categorical(col, X[col]):
                x_clean = X[col].dropna().astype(str)
                y_clean = y.loc[x_clean.index].astype(str)
                univariate[col] = cramers_v(x_clean, y_clean)
            else:
                univariate[col] = abs(point_biserial(X[col], y))
        except Exception as e:
            univariate[col] = 0.0
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[6] Imputing (most-frequent) for MI + RF ...")
    t0 = time.time()
    imputer = SimpleImputer(strategy="most_frequent")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_usable),
        columns=usable_cols,
        index=X_usable.index,
    )
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[7] Mutual information ...")
    t0 = time.time()
    discrete = [is_categorical(c, X[c]) for c in usable_cols]
    mi = mutual_info_classif(
        X_imp.values, y.values,
        discrete_features=discrete, random_state=SEED, n_jobs=-1,
    )
    mi_dict = {c: float(s) for c, s in zip(usable_cols, mi)}
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[8] RandomForest importance (single fit, 300 trees) ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, n_jobs=-1, random_state=SEED
    )
    rf.fit(X_imp.values, y.values)
    rf_dict = {c: float(s) for c, s in zip(usable_cols, rf.feature_importances_)}
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[9] Writing feature_audit.csv ...")
    rows = []
    for col in feature_cols:
        rows.append({
            "name": col,
            "tier": tiers[col],
            "rf_importance": rf_dict.get(col, np.nan),
            "univariate": univariate.get(col, np.nan),
            "mi": mi_dict.get(col, np.nan),
        })
    audit = pd.DataFrame(rows)
    audit = audit.sort_values("rf_importance", ascending=False).reset_index(drop=True)
    audit.to_csv(OUT_DIR / "feature_audit.csv", index=False)
    print(f"  wrote {OUT_DIR / 'feature_audit.csv'}  ({len(audit)} rows)")

    print("[10] Writing feature_audit.md ...")
    write_audit_md(audit, OUT_DIR / "feature_audit.md")
    print(f"  wrote {OUT_DIR / 'feature_audit.md'}")

    print("\n[11] Target evaluation (4 candidates, LR baseline 3-fold) ...")
    candidates = ["ADDEPEV3", "_MENT14D", "MENTHLTH_b14", "LSATISFY_dissat"]
    # Same `core+ses+aces` features as the modeling script (use raw, no engineered ace_score)
    feats_all = [
        "_AGEG5YR", "_SEX", "_BMI5", "MARITAL", "EDUCA", "SMOKE100", "DRNKANY6",
        "_INCOMG1", "EMPLOY1", "GENHLTH", "_HLTHPL2",
        "LSATISFY", "SDLONELY",
        "DIABETE4", "CVDINFR4", "ASTHMA3", "HAVARTH4",
    ]
    # Source variables that would leak each candidate target.
    leakage_map = {
        "ADDEPEV3": set(),
        "_MENT14D": {"MENTHLTH", "POORHLTH", "_RFHLTH"},
        "MENTHLTH_b14": {"MENTHLTH", "POORHLTH", "_RFHLTH"},
        "LSATISFY_dissat": {"LSATISFY"},
    }
    target_results = []
    for cand in candidates:
        t0 = time.time()
        sub_feats = [f for f in feats_all if f not in leakage_map[cand]]
        try:
            res = evaluate_target(cand, df_sub, sub_feats)
            res["n_features"] = len(sub_feats)
        except Exception as e:
            res = {"target": cand, "prevalence": np.nan, "missingness": np.nan,
                   "n_valid": 0, "baseline_roc_auc": np.nan, "baseline_f1": np.nan,
                   "n_features": 0}
            print(f"  {cand}: FAILED ({e})")
        res["time_s"] = time.time() - t0
        target_results.append(res)
        print(
            f"  {cand:>20s}  prev={res['prevalence']:.3f}  miss={res['missingness']:.3f}  "
            f"auc={res['baseline_roc_auc']:.3f}  f1={res['baseline_f1']:.3f}  "
            f"n_feats={res['n_features']}  ({res['time_s']:.1f}s)"
        )
    targets_df = pd.DataFrame(target_results)
    targets_df.to_csv(OUT_DIR / "target_comparison.csv", index=False)
    print(f"  wrote {OUT_DIR / 'target_comparison.csv'}")

    # Cross-tab vs ADDEPEV3 for the other 3 targets
    print("\n[12] Cross-tab vs ADDEPEV3 ...")
    cross_tabs = {}
    y_addepev3 = (df_sub["ADDEPEV3"] if "ADDEPEV3" in df_sub.columns else
                  # Reconstruct from current target
                  df_sub["target"])
    for cand in ["_MENT14D", "MENTHLTH_b14", "LSATISFY_dissat"]:
        y_cand = make_target(cand, df_sub)
        valid = y_cand.notna() & y_addepev3.notna()
        ct = pd.crosstab(
            y_addepev3.loc[valid].astype(int),
            y_cand.loc[valid].astype(int),
            margins=True,
        )
        cross_tabs[cand] = ct
        print(f"\n  {cand} (rows=ADDEPEV3, cols={cand}):")
        print(ct.to_string())

    # Write target md
    write_target_md(targets_df, OUT_DIR / "target_comparison.md")

    # Append cross-tabs to the target md
    extra = ["\n## Cross-tabulations\n"]
    for cand, ct in cross_tabs.items():
        extra.append(f"### {cand} vs ADDEPEV3 (rows=ADDEPEV3, cols={cand})\n")
        extra.append("```\n" + ct.to_string() + "\n```\n")
    with open(OUT_DIR / "target_comparison.md", "a", encoding="utf-8") as f:
        f.write("\n".join(extra))

    print(f"\n[done] wrote {OUT_DIR / 'target_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
