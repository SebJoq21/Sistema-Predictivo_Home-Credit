"""
3.5.1 Preprocesamiento de `bureau` y `bureau_balance` con agregación en 2 niveles
(`SCRUM-20`).
Nivel 1: `bureau_balance` -> agregado por `SK_ID_BUREAU` (meses observados, meses en mora STATUS 1-5, proporción cerrada/activa).
Nivel 2: `bureau` + agregados de `bureau_balance` -> agregado por `SK_ID_CURR` (solo clientes de `application_train`), creando la bandera `TIENE_BUREAU`.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.memory_optimization import read_csv_optimized, reduce_mem_usage

RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"


def run_agg_bureau(train_ids: set[int] | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if train_ids is None:
        train_ids = set(
            pd.read_csv(RAW_DIR / "application_train.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"]
        )

    print("[agg_bureau] 1/3 Leyendo bureau.csv filtrado a SK_ID_CURR de train...")
    bureau = read_csv_optimized(
        RAW_DIR / "bureau.csv",
        filter_col="SK_ID_CURR",
        valid_ids=train_ids,
        chunksize=1_000_000,
    )
    valid_bureau_ids = set(bureau["SK_ID_BUREAU"].unique())

    print("[agg_bureau] 2/3 Agregando bureau_balance.csv por bloques (Nivel 1: SK_ID_BUREAU)...")
    bb_chunks = []
    for chunk in pd.read_csv(RAW_DIR / "bureau_balance.csv", chunksize=2_000_000):
        chunk = chunk[chunk["SK_ID_BUREAU"].isin(valid_bureau_ids)].copy()
        if len(chunk) == 0:
            continue
        status_str = chunk["STATUS"].astype(str)
        chunk["BB_DPD_MONTH"] = status_str.isin(["1", "2", "3", "4", "5"]).astype("int8")
        chunk["BB_SEVERE_DPD"] = status_str.isin(["3", "4", "5"]).astype("int8")
        chunk["BB_CLOSED"] = (status_str == "C").astype("int8")

        g = chunk.groupby("SK_ID_BUREAU", observed=False).agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
            BB_MONTHS_MAX=("MONTHS_BALANCE", "max"),
            BB_DPD_MONTHS_SUM=("BB_DPD_MONTH", "sum"),
            BB_SEVERE_DPD_SUM=("BB_SEVERE_DPD", "sum"),
            BB_CLOSED_SUM=("BB_CLOSED", "sum"),
        )
        bb_chunks.append(g)

    if bb_chunks:
        bb_all = pd.concat(bb_chunks).groupby("SK_ID_BUREAU", observed=False).agg(
            BB_MONTHS_COUNT=("BB_MONTHS_COUNT", "sum"),
            BB_MONTHS_MIN=("BB_MONTHS_MIN", "min"),
            BB_MONTHS_MAX=("BB_MONTHS_MAX", "max"),
            BB_DPD_MONTHS_SUM=("BB_DPD_MONTHS_SUM", "sum"),
            BB_SEVERE_DPD_SUM=("BB_SEVERE_DPD_SUM", "sum"),
            BB_CLOSED_SUM=("BB_CLOSED_SUM", "sum"),
        ).reset_index()
        bb_all["BB_DPD_RATIO"] = (bb_all["BB_DPD_MONTHS_SUM"] / bb_all["BB_MONTHS_COUNT"]).astype("float32")
    else:
        bb_all = pd.DataFrame(columns=["SK_ID_BUREAU", "BB_MONTHS_COUNT", "BB_DPD_MONTHS_SUM", "BB_SEVERE_DPD_SUM", "BB_DPD_RATIO"])

    del bb_chunks
    gc.collect()

    print("[agg_bureau] 3/3 Uniendo Nivel 1 con bureau y agregando por SK_ID_CURR (Nivel 2)...")
    bureau = bureau.merge(bb_all, on="SK_ID_BUREAU", how="left")
    del bb_all
    gc.collect()

    cred_active = bureau["CREDIT_ACTIVE"].astype(str)
    bureau["IS_ACTIVE"] = (cred_active == "Active").astype("int8")
    bureau["IS_CLOSED"] = (cred_active == "Closed").astype("int8")
    bureau["IS_BAD_DEBT"] = cred_active.isin(["Bad debt", "Sold"]).astype("int8")
    bureau["DEBT_CREDIT_RATIO"] = (
        bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"].replace(0, np.nan))
    ).astype("float32")

    bureau_agg = bureau.groupby("SK_ID_CURR", observed=False).agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_CLOSED_COUNT=("IS_CLOSED", "sum"),
        BUREAU_BAD_DEBT_COUNT=("IS_BAD_DEBT", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUREAU_DAYS_ENDDATE_MEAN=("DAYS_CREDIT_ENDDATE", "mean"),
        BUREAU_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_AMT_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_AMT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_DEBT_SUM_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_DEBT_SUM_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_DEBT_RATIO_MEAN=("DEBT_CREDIT_RATIO", "mean"),
        BUREAU_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        BUREAU_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_BB_MONTHS_TOTAL=("BB_MONTHS_COUNT", "sum"),
        BUREAU_BB_DPD_MONTHS_SUM=("BB_DPD_MONTHS_SUM", "sum"),
        BUREAU_BB_SEVERE_DPD_SUM=("BB_SEVERE_DPD_SUM", "sum"),
        BUREAU_BB_DPD_RATIO_MEAN=("BB_DPD_RATIO", "mean"),
    ).reset_index()

    bureau_agg["BUREAU_ACTIVE_RATIO"] = (
        bureau_agg["BUREAU_ACTIVE_COUNT"] / bureau_agg["BUREAU_CREDIT_COUNT"]
    ).astype("float32")
    bureau_agg["BUREAU_TOTAL_DEBT_TO_CREDIT"] = (
        bureau_agg["BUREAU_DEBT_SUM_TOTAL"] / bureau_agg["BUREAU_CREDIT_SUM_TOTAL"].replace(0, np.nan)
    ).astype("float32")

    bureau_agg = reduce_mem_usage(bureau_agg, cat_exclude=["SK_ID_CURR"])
    out_path = PROCESSED_DIR / "bureau_agg.parquet"
    bureau_agg.to_parquet(out_path, index=False)
    print(f"[agg_bureau] Guardado {out_path} ({bureau_agg.shape})")

    del bureau
    gc.collect()
    return bureau_agg


if __name__ == "__main__":
    run_agg_bureau()
