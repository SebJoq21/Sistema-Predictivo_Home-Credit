"""
3.2 Estrategia de lectura por bloques y tipos de datos reducidos (`SCRUM-17`).
Provee funciones de carga optimizada de las 7 tablas de Home Credit usando
`utils.memory_optimization`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.memory_optimization import read_csv_optimized, reduce_mem_usage

RAW_DIR = Path("data") / "raw"
SAMPLE_IDS_PATH = Path("data") / "sample" / "sample_ids.csv"


def get_target_ids(sample: bool = False) -> pd.Series:
    """Devuelve la serie de SK_ID_CURR de train (100% o muestra 10%)."""
    if sample and SAMPLE_IDS_PATH.exists():
        return pd.read_csv(SAMPLE_IDS_PATH, usecols=["SK_ID_CURR"])["SK_ID_CURR"]
    return pd.read_csv(RAW_DIR / "application_train.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"]


def load_table(
    table_name: str,
    usecols: Optional[Iterable[str]] = None,
    filter_col: Optional[str] = "SK_ID_CURR",
    valid_ids: Optional[Iterable[int]] = None,
    chunksize: int = 1_000_000,
    downcast: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Carga una tabla desde `data/raw/{table_name}.csv` por bloques y reduce memoria."""
    csv_path = RAW_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    return read_csv_optimized(
        filepath=csv_path,
        usecols=usecols,
        filter_col=filter_col,
        valid_ids=valid_ids,
        chunksize=chunksize,
        downcast=downcast,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de carga optimizada de tablas.")
    parser.add_argument("--sample", action="store_true", help="Usar muestra 10%")
    parser.add_argument("--full", action="store_true", help="Usar 100% de train")
    args = parser.parse_args()

    ids = get_target_ids(sample=args.sample and not args.full)
    df_app = load_table("application_train", valid_ids=ids)
    print(f"[load_data] application_train cargado: {df_app.shape}")
