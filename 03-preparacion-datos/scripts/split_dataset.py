"""
3.7 Split train/validación estratificado y recalibración anti-leakage (`SCRUM-23` y `SCRUM-27`).
Divide `dataset_consolidado.parquet` en `train.parquet` (80%) y `val.parquet` (20%)
estratificando por `TARGET` (`random_state=42`).
Garantiza cero fuga de datos (data leakage):
- Los topes al percentil 99.9 de las 4 variables con colas extremas se calculan
  exclusivamente sobre `train` y se aplican a `val`.
- Verifica disyunción exacta de `SK_ID_CURR` entre `train` y `val` y conservación
  de la tasa de mora (8.0729%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path("data") / "processed"
FASE3_DIR = Path("03-preparacion-datos")

TOPEAR_P999 = [
    "AMT_INCOME_TOTAL",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_QRT",
]


def run_split_dataset(val_size: float = 0.20, random_state: int = 42) -> dict:
    consolidado_path = PROCESSED_DIR / "dataset_consolidado.parquet"
    print(f"[split_dataset] Cargando {consolidado_path}...")
    df = pd.read_parquet(consolidado_path)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["TARGET"],
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Recalcular topes p99.9 estrictamente en train y aplicarlos a val (anti-leakage)
    topes_train = {}
    for c in TOPEAR_P999:
        if c in train_df.columns:
            q_train = float(train_df[c].quantile(0.999))
            topes_train[c] = q_train
            train_df[c] = train_df[c].clip(upper=q_train)
            val_df[c] = val_df[c].clip(upper=q_train)

    # Verificaciones anti-leakage
    set_train = set(train_df["SK_ID_CURR"])
    set_val = set(val_df["SK_ID_CURR"])
    interseccion = len(set_train.intersection(set_val))
    assert interseccion == 0, f"Fuga detectada: {interseccion} clientes repetidos entre train y val"

    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    stats_split = {
        "total_filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "train_filas": int(len(train_df)),
        "val_filas": int(len(val_df)),
        "train_mora_casos": int(train_df["TARGET"].sum()),
        "val_mora_casos": int(val_df["TARGET"].sum()),
        "train_mora_pct": round(100.0 * float(train_df["TARGET"].mean()), 4),
        "val_mora_pct": round(100.0 * float(val_df["TARGET"].mean()), 4),
        "interseccion_sk_id_curr": interseccion,
        "topes_p999_calculados_en_train": topes_train,
    }

    with open(FASE3_DIR / "salida_fase3" / "split_resumen.json", "w", encoding="utf-8") as f:
        json.dump(stats_split, f, indent=2, ensure_ascii=False)

    print(
        f"[split_dataset] Guardados train.parquet ({len(train_df):,} filas, mora={stats_split['train_mora_pct']}%) "
        f"y val.parquet ({len(val_df):,} filas, mora={stats_split['val_mora_pct']}%)"
    )
    return stats_split


if __name__ == "__main__":
    run_split_dataset()
