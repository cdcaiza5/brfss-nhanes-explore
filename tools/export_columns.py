from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat

# tools/export_columns.py -> tools/ -> raiz
REPO_DIR = Path(__file__).resolve().parent.parent
XPT_PATH = REPO_DIR / "data" / "LLCP2024.XPT_"
OUT_DIR = REPO_DIR / "metadata" / "columns"
CSV_PATH = OUT_DIR / "brfss_2024_columns.csv"
MD_PATH = OUT_DIR / "brfss_2024_columns.md"


def build_table() -> pd.DataFrame:
    """Lee el XPT y devuelve un DataFrame con position, name, label para las 301 columnas.

    Returns:
        DataFrame con 301 filas y columnas position, name, label.
    """
    _, meta = pyreadstat.read_xport(str(XPT_PATH), encoding="cp1252")
    rows = []
    for i, (name, label) in enumerate(
        zip(meta.column_names, meta.column_labels), start=1
    ):
        rows.append(
            {
                "position": i,
                "name": name,
                "label": "" if label is None else str(label),
            }
        )
    return pd.DataFrame(rows, columns=["position", "name", "label"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Escribe el DataFrame como CSV UTF-8 sin indice.

    Args:
        df: tabla a escribir.
        path: ruta destino.
    """
    df.to_csv(path, index=False, encoding="utf-8")


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    """Escribe el DataFrame como tabla markdown UTF-8.

    Args:
        df: tabla a escribir.
        path: ruta destino.
    """
    headers = ["position", "name", "label"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in df.itertuples(index=False, name=None):
        cells = [
            str(c).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            for c in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Punto de entrada. Crea el directorio de salida y genera CSV y MD."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table()
    write_csv(df, CSV_PATH)
    write_markdown(df, MD_PATH)
    # ponytail: el XPT tiene 301 columnas y 2 vars sin label; el check es
    # barato y evita sobre-escribir con una tabla rota.
    assert len(df) == 301, f"expected 301 rows, got {len(df)}"
    assert (df["label"] == "").sum() == 2
    print(f"rows: {len(df)}  (vars with empty label: {(df['label'] == '').sum()})")
    print(f"csv: {CSV_PATH}  ({CSV_PATH.stat().st_size} bytes)")
    print(f"md:  {MD_PATH}  ({MD_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
