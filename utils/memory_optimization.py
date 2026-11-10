"""
Utilidades de optimización de memoria y lectura por bloques (CRISP-DM Fase 2/3).
Cumple el requisito de Gobierno de Datos Analíticos del Business Case y Project Charter:
downcasting de tipos numéricos (float64 -> float32, int64 -> int8/int16/int32),
conversión de texto a category y recolección de basura explícita (gc.collect()).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def memory_mb(df: pd.DataFrame) -> float:
    """Devuelve el uso de memoria del DataFrame en MB (incluyendo objetos/categorías)."""
    return round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 2)


def reduce_mem_usage(
    df: pd.DataFrame,
    cat_exclude: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reduce el consumo de RAM de un DataFrame mediante downcasting seguro:
    - Columnas de texto (object/str/string) pasan a `category`.
    - Enteros se reducen a int8 / int16 / int32 según su rango.
    - Flotantes float64 pasan a float32 preservando NaN.
    """
    exclude = set(cat_exclude or [])
    start_mem = memory_mb(df)

    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        dtype = s.dtype

        if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            continue

        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            df[col] = s.astype("category")
        elif pd.api.types.is_integer_dtype(dtype):
            c_min, c_max = int(s.min()), int(s.max())
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = s.astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = s.astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = s.astype(np.int32)
            else:
                df[col] = s.astype(np.int64)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(s, downcast="float")

    end_mem = memory_mb(df)
    if verbose and start_mem > 0:
        pct_red = 100.0 * (start_mem - end_mem) / start_mem
        print(
            f"[memory_optimization] RAM reducida: {start_mem:.2f} MB -> {end_mem:.2f} MB "
            f"(-{pct_red:.1f}%)"
        )
    return df


def read_csv_optimized(
    filepath: str | Path,
    usecols: Optional[Iterable[str]] = None,
    filter_col: Optional[str] = None,
    valid_ids: Optional[set | np.ndarray | pd.Series] = None,
    chunksize: int = 1_000_000,
    downcast: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lee un archivo CSV masivo por bloques (`chunksize`), filtrando opcionalmente
    por un conjunto de llaves válidas (ej. `SK_ID_CURR` de `application_train`)
    antes de concatenar en memoria y aplicando `reduce_mem_usage`.
    """
    filepath = Path(filepath)
    valid_set = set(valid_ids) if valid_ids is not None else None
    chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(filepath, usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        if filter_col and valid_set is not None and filter_col in chunk.columns:
            chunk = chunk[chunk[filter_col].isin(valid_set)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        df = pd.read_csv(filepath, usecols=usecols, nrows=0)
    else:
        df = pd.concat(chunks, ignore_index=True)

    del chunks
    gc.collect()

    if downcast and len(df) > 0:
        df = reduce_mem_usage(df, verbose=verbose)

    return df
