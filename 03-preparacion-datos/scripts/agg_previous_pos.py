"""
3.5.2 Preprocesamiento de `previous_application` y `POS_CASH_balance` (`SCRUM-21`).
Excluye `RATE_INTEREST_PRIMARY` y `RATE_INTEREST_PRIVILEGED` (99.64% nulos),
limpia el centinela `365243` en `DAYS_*` de solicitudes previas y agrega ambas tablas
a nivel de cliente (`SK_ID_CURR` de `application_train`).
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
SENTINELA = 365243


def run_agg_previous_pos(train_ids: set[int] | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if train_ids is None:
        train_ids = set(
            pd.read_csv(RAW_DIR / "application_train.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"]
        )

    print("[agg_previous_pos] 1/2 Procesando previous_application.csv...")
    prev_cols = [
        "SK_ID_PREV",
        "SK_ID_CURR",
        "NAME_CONTRACT_TYPE",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "NAME_CONTRACT_STATUS",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
        "NFLAG_INSURED_ON_APPROVAL",
    ]
    prev = read_csv_optimized(
        RAW_DIR / "previous_application.csv",
        usecols=prev_cols,
        filter_col="SK_ID_CURR",
        valid_ids=train_ids,
        chunksize=1_000_000,
    )

    for dcol in [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]:
        prev.loc[prev[dcol] == SENTINELA, dcol] = np.nan

    status_str = prev["NAME_CONTRACT_STATUS"].astype(str)
    prev["PREV_IS_APPROVED"] = (status_str == "Approved").astype("int8")
    prev["PREV_IS_REFUSED"] = (status_str == "Refused").astype("int8")
    prev["PREV_IS_CANCELED"] = (status_str == "Canceled").astype("int8")
    prev["PREV_APP_CREDIT_RATIO"] = (
        prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)
    ).astype("float32")

    prev_agg = prev.groupby("SK_ID_CURR", observed=False).agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("PREV_IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("PREV_IS_REFUSED", "sum"),
        PREV_CANCELED_COUNT=("PREV_IS_CANCELED", "sum"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
        PREV_APP_CREDIT_RATIO_MEAN=("PREV_APP_CREDIT_RATIO", "mean"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_INSURED_MEAN=("NFLAG_INSURED_ON_APPROVAL", "mean"),
    ).reset_index()

    prev_agg["PREV_APPROVAL_RATE"] = (
        prev_agg["PREV_APPROVED_COUNT"] / prev_agg["PREV_APP_COUNT"]
    ).astype("float32")
    prev_agg["PREV_REFUSAL_RATE"] = (
        prev_agg["PREV_REFUSED_COUNT"] / prev_agg["PREV_APP_COUNT"]
    ).astype("float32")

    del prev
    gc.collect()

    print("[agg_previous_pos] 2/2 Procesando POS_CASH_balance.csv por bloques...")
    pos_chunks = []
    for chunk in pd.read_csv(RAW_DIR / "POS_CASH_balance.csv", chunksize=1_500_000):
        chunk = chunk[chunk["SK_ID_CURR"].isin(train_ids)].copy()
        if len(chunk) == 0:
            continue
        chunk["POS_HAS_DPD"] = (chunk["SK_DPD"] > 0).astype("int8")
        chunk["POS_HAS_DPD_DEF"] = (chunk["SK_DPD_DEF"] > 0).astype("int8")
        chunk["POS_IS_COMPLETED"] = (chunk["NAME_CONTRACT_STATUS"].astype(str) == "Completed").astype("int8")

        g = chunk.groupby("SK_ID_CURR", observed=False).agg(
            POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            POS_DPD_MAX=("SK_DPD", "max"),
            POS_DPD_SUM=("SK_DPD", "sum"),
            POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            POS_DPD_DEF_SUM=("SK_DPD_DEF", "sum"),
            POS_DPD_MONTHS_SUM=("POS_HAS_DPD", "sum"),
            POS_DPD_DEF_MONTHS_SUM=("POS_HAS_DPD_DEF", "sum"),
            POS_COMPLETED_SUM=("POS_IS_COMPLETED", "sum"),
            POS_INSTALMENT_FUTURE_SUM=("CNT_INSTALMENT_FUTURE", "sum"),
        )
        pos_chunks.append(g)

    pos_agg = pd.concat(pos_chunks).groupby("SK_ID_CURR", observed=False).agg(
        POS_MONTHS_COUNT=("POS_MONTHS_COUNT", "sum"),
        POS_DPD_MAX=("POS_DPD_MAX", "max"),
        POS_DPD_SUM=("POS_DPD_SUM", "sum"),
        POS_DPD_DEF_MAX=("POS_DPD_DEF_MAX", "max"),
        POS_DPD_DEF_SUM=("POS_DPD_DEF_SUM", "sum"),
        POS_DPD_MONTHS_SUM=("POS_DPD_MONTHS_SUM", "sum"),
        POS_DPD_DEF_MONTHS_SUM=("POS_DPD_DEF_MONTHS_SUM", "sum"),
        POS_COMPLETED_SUM=("POS_COMPLETED_SUM", "sum"),
        POS_INSTALMENT_FUTURE_SUM=("POS_INSTALMENT_FUTURE_SUM", "sum"),
    ).reset_index()

    pos_agg["POS_DPD_MEAN"] = (pos_agg["POS_DPD_SUM"] / pos_agg["POS_MONTHS_COUNT"]).astype("float32")
    pos_agg["POS_DPD_DEF_MEAN"] = (pos_agg["POS_DPD_DEF_SUM"] / pos_agg["POS_MONTHS_COUNT"]).astype("float32")
    pos_agg["POS_DPD_MONTHS_RATIO"] = (pos_agg["POS_DPD_MONTHS_SUM"] / pos_agg["POS_MONTHS_COUNT"]).astype("float32")

    del pos_chunks
    gc.collect()

    merged = pd.merge(prev_agg, pos_agg, on="SK_ID_CURR", how="outer")
    merged = reduce_mem_usage(merged, cat_exclude=["SK_ID_CURR"])

    out_path = PROCESSED_DIR / "previous_pos_agg.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[agg_previous_pos] Guardado {out_path} ({merged.shape})")

    del prev_agg, pos_agg
    gc.collect()
    return merged


if __name__ == "__main__":
    run_agg_previous_pos()
