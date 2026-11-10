"""
3.5.3 Preprocesamiento de `credit_card_balance` e `installments_payments` (`SCRUM-22`).
Calcula tendencias de pago en cuotas (DPD, pago anticipado/tardío, fracción pagada vs exigida)
y utilización de línea de tarjeta de crédito por cliente (`SK_ID_CURR` de `application_train`).
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

from utils.memory_optimization import reduce_mem_usage

RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"


def run_agg_cards(train_ids: set[int] | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if train_ids is None:
        train_ids = set(
            pd.read_csv(RAW_DIR / "application_train.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"]
        )

    print("[agg_cards] 1/2 Procesando installments_payments.csv por bloques...")
    inst_chunks = []
    for chunk in pd.read_csv(RAW_DIR / "installments_payments.csv", chunksize=1_500_000):
        chunk = chunk[chunk["SK_ID_CURR"].isin(train_ids)].copy()
        if len(chunk) == 0:
            continue
        dpd = (chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]).clip(lower=0)
        dbd = (chunk["DAYS_INSTALMENT"] - chunk["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
        diff = chunk["AMT_INSTALMENT"] - chunk["AMT_PAYMENT"]

        chunk["INST_DPD"] = dpd
        chunk["INST_DBD"] = dbd
        chunk["INST_IS_LATE"] = (dpd > 0).astype("int8")
        chunk["INST_IS_UNDERPAID"] = (diff > 1e-2).astype("int8")
        chunk["INST_PAYMENT_DIFF"] = diff

        g = chunk.groupby("SK_ID_CURR", observed=False).agg(
            INST_COUNT=("SK_ID_PREV", "count"),
            INST_DPD_MAX=("INST_DPD", "max"),
            INST_DPD_SUM=("INST_DPD", "sum"),
            INST_DBD_SUM=("INST_DBD", "sum"),
            INST_LATE_COUNT=("INST_IS_LATE", "sum"),
            INST_UNDERPAID_COUNT=("INST_IS_UNDERPAID", "sum"),
            INST_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
            INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
            INST_AMT_PAYMENT_MAX=("AMT_PAYMENT", "max"),
            INST_PAYMENT_DIFF_MAX=("INST_PAYMENT_DIFF", "max"),
            INST_VERSION_MAX=("NUM_INSTALMENT_VERSION", "max"),
        )
        inst_chunks.append(g)

    inst_agg = pd.concat(inst_chunks).groupby("SK_ID_CURR", observed=False).agg(
        INST_COUNT=("INST_COUNT", "sum"),
        INST_DPD_MAX=("INST_DPD_MAX", "max"),
        INST_DPD_SUM=("INST_DPD_SUM", "sum"),
        INST_DBD_SUM=("INST_DBD_SUM", "sum"),
        INST_LATE_COUNT=("INST_LATE_COUNT", "sum"),
        INST_UNDERPAID_COUNT=("INST_UNDERPAID_COUNT", "sum"),
        INST_AMT_INSTALMENT_SUM=("INST_AMT_INSTALMENT_SUM", "sum"),
        INST_AMT_PAYMENT_SUM=("INST_AMT_PAYMENT_SUM", "sum"),
        INST_AMT_PAYMENT_MAX=("INST_AMT_PAYMENT_MAX", "max"),
        INST_PAYMENT_DIFF_MAX=("INST_PAYMENT_DIFF_MAX", "max"),
        INST_VERSION_MAX=("INST_VERSION_MAX", "max"),
    ).reset_index()

    inst_agg["INST_DPD_MEAN"] = (inst_agg["INST_DPD_SUM"] / inst_agg["INST_COUNT"]).astype("float32")
    inst_agg["INST_DBD_MEAN"] = (inst_agg["INST_DBD_SUM"] / inst_agg["INST_COUNT"]).astype("float32")
    inst_agg["INST_LATE_RATIO"] = (inst_agg["INST_LATE_COUNT"] / inst_agg["INST_COUNT"]).astype("float32")
    inst_agg["INST_UNDERPAID_RATIO"] = (inst_agg["INST_UNDERPAID_COUNT"] / inst_agg["INST_COUNT"]).astype("float32")
    inst_agg["INST_PAYMENT_TO_INSTALMENT_RATIO"] = (
        inst_agg["INST_AMT_PAYMENT_SUM"] / inst_agg["INST_AMT_INSTALMENT_SUM"].replace(0, np.nan)
    ).astype("float32")

    del inst_chunks
    gc.collect()

    print("[agg_cards] 2/2 Procesando credit_card_balance.csv por bloques...")
    cc_chunks = []
    for chunk in pd.read_csv(RAW_DIR / "credit_card_balance.csv", chunksize=1_000_000):
        chunk = chunk[chunk["SK_ID_CURR"].isin(train_ids)].copy()
        if len(chunk) == 0:
            continue
        chunk["CC_UTILIZATION"] = (
            chunk["AMT_BALANCE"] / chunk["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        )
        chunk["CC_HAS_DPD"] = (chunk["SK_DPD"] > 0).astype("int8")

        g = chunk.groupby("SK_ID_CURR", observed=False).agg(
            CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
            CC_BALANCE_MAX=("AMT_BALANCE", "max"),
            CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_UTILIZATION_MEAN=("CC_UTILIZATION", "mean"),
            CC_UTILIZATION_MAX=("CC_UTILIZATION", "max"),
            CC_DRAWINGS_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
            CC_DRAWINGS_ATM_SUM=("AMT_DRAWINGS_ATM_CURRENT", "sum"),
            CC_PAYMENT_TOTAL_SUM=("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
            CC_DPD_MAX=("SK_DPD", "max"),
            CC_DPD_SUM=("SK_DPD", "sum"),
            CC_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            CC_DPD_MONTHS_SUM=("CC_HAS_DPD", "sum"),
        )
        cc_chunks.append(g)

    cc_agg = pd.concat(cc_chunks).groupby("SK_ID_CURR", observed=False).agg(
        CC_MONTHS_COUNT=("CC_MONTHS_COUNT", "sum"),
        CC_BALANCE_MEAN=("CC_BALANCE_MEAN", "mean"),
        CC_BALANCE_MAX=("CC_BALANCE_MAX", "max"),
        CC_LIMIT_MEAN=("CC_LIMIT_MEAN", "mean"),
        CC_UTILIZATION_MEAN=("CC_UTILIZATION_MEAN", "mean"),
        CC_UTILIZATION_MAX=("CC_UTILIZATION_MAX", "max"),
        CC_DRAWINGS_SUM=("CC_DRAWINGS_SUM", "sum"),
        CC_DRAWINGS_ATM_SUM=("CC_DRAWINGS_ATM_SUM", "sum"),
        CC_PAYMENT_TOTAL_SUM=("CC_PAYMENT_TOTAL_SUM", "sum"),
        CC_DPD_MAX=("CC_DPD_MAX", "max"),
        CC_DPD_SUM=("CC_DPD_SUM", "sum"),
        CC_DPD_DEF_MAX=("CC_DPD_DEF_MAX", "max"),
        CC_DPD_MONTHS_SUM=("CC_DPD_MONTHS_SUM", "sum"),
    ).reset_index()

    cc_agg["CC_DPD_RATIO"] = (cc_agg["CC_DPD_MONTHS_SUM"] / cc_agg["CC_MONTHS_COUNT"]).astype("float32")
    del cc_chunks
    gc.collect()

    merged = pd.merge(inst_agg, cc_agg, on="SK_ID_CURR", how="outer")
    merged = reduce_mem_usage(merged, cat_exclude=["SK_ID_CURR"])

    out_path = PROCESSED_DIR / "cc_installments_agg.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[agg_cards] Guardado {out_path} ({merged.shape})")

    del inst_agg, cc_agg
    gc.collect()
    return merged


if __name__ == "__main__":
    run_agg_cards()
