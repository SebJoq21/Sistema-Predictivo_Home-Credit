"""
3.1 Muestra estratificada del 10% (contingencia del Project Charter).
Genera `data/sample/sample_ids.csv` (30.751 clientes estratificados por TARGET,
random_state=42) para validaciones rápidas del pipeline.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path("data") / "raw"
SAMPLE_DIR = Path("data") / "sample"


def generate_sample_ids(sample_frac: float = 0.10, random_state: int = 42) -> Path:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    app_path = RAW_DIR / "application_train.csv"
    df_ids = pd.read_csv(app_path, usecols=["SK_ID_CURR", "TARGET"])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=sample_frac, random_state=random_state
    )
    _, sample_idx = next(splitter.split(df_ids, df_ids["TARGET"]))
    sample_df = (
        df_ids.iloc[sample_idx]
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )

    out_path = SAMPLE_DIR / "sample_ids.csv"
    sample_df.to_csv(out_path, index=False)
    mora_pct = 100.0 * sample_df["TARGET"].mean()
    print(
        f"[make_sample] Guardado {out_path} ({len(sample_df):,} filas, "
        f"mora={mora_pct:.2f}%)"
    )
    return out_path


if __name__ == "__main__":
    generate_sample_ids()
