"""
Cargador estandarizado de datos y metadatos para la Fase 4: Modelado (CRISP-DM).
Garantiza:
1. Uso exclusivo de las 240 variables autorizadas (`incluir_modelo == True` en `lista_features_final.csv`).
2. Exclusión estricta de `SK_ID_CURR`, `TARGET` y variables protegidas (`CODE_GENDER`,
   `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`) en cumplimiento de la Ley N.° 29733 y normativa SBS.
3. Conservación de tipos optimizados (`float32`, `int8/16`, `category`) y categorías consistentes entre train y val.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_IDS_PATH = ROOT / "data" / "sample" / "sample_ids.csv"
FEATURES_CSV_PATH = ROOT / "03-preparacion-datos" / "lista_features_final.csv"

PROTECTED_AND_STRUCTURAL = {
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
}


def classify_feature_family(col: str) -> str:
    """Clasifica cada variable predictora según su dominio u origen relacional."""
    if "EXT_SOURCE" in col:
        return "Scores Externos (EXT_SOURCE)"
    if col.startswith("INST_"):
        return "Pagos de Cuotas (installments)"
    if col.startswith("CC_") or col == "TIENE_TARJETA":
        return "Tarjeta de Crédito (credit_card)"
    if col.startswith("BUREAU_") or col == "TIENE_BUREAU" or "CREDIT_BUREAU" in col:
        return "Buró Externo (bureau + balance)"
    if col.startswith("PREV_") or col.startswith("POS_") or col == "TIENE_PREV_APP" or col.startswith("HIST_"):
        return "Solicitudes Previas y POS_CASH"
    if any(
        col.startswith(p)
        for p in (
            "CREDIT_TO_",
            "ANNUITY_TO_",
            "INCOME_PER_",
            "TOTAL_DEBT_TO_",
            "AMT_",
            "TOPEADO_AMT_",
        )
    ):
        return "Ratios Financieros y Montos"
    if col.startswith("FALTA_") or any(
        k in col
        for k in (
            "APARTMENTS",
            "AREA",
            "ELEVATORS",
            "ENTRANCES",
            "FLOORS",
            "YEARS_BUILD",
            "YEARS_BEGIN",
            "HOUSETYPE",
            "WALLSMATERIAL",
            "FONDKAPREMONT",
            "EMERGENCYSTATE",
        )
    ):
        return "Vivienda y Banderas de Ausencia"
    return "Demográficas, Empleo y Estabilidad"


def get_authorized_features() -> Tuple[list[str], list[str], list[str], pd.DataFrame]:
    """
    Lee `lista_features_final.csv` y retorna:
    - `feature_cols`: las 240 variables con `incluir_modelo == True`
    - `num_cols`: subconjunto numérico
    - `cat_cols`: subconjunto categórico (`category`)
    - `df_feat`: DataFrame completo del catálogo
    """
    df_feat = pd.read_csv(FEATURES_CSV_PATH)
    sel = df_feat[df_feat["incluir_modelo"] == True].copy()  # noqa: E712
    feature_cols = sel["variable"].tolist()

    forbidden_found = PROTECTED_AND_STRUCTURAL.intersection(feature_cols)
    assert not forbidden_found, f"Violación de gobernanza: columnas prohibidas en features: {forbidden_found}"
    assert len(feature_cols) == 240, f"Se esperaban 240 variables predictoras, se encontraron {len(feature_cols)}"

    cat_cols = sel[sel["tipo"] == "category"]["variable"].tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols, df_feat


def load_train_val(use_sample: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Carga `train.parquet` y `val.parquet`, alinea los dominios de las columnas `category`
    usando estrictamente las categorías vistas en `train`, y retorna:
    `(X_train, y_train, X_val, y_val, num_cols, cat_cols)`
    """
    feature_cols, num_cols, cat_cols, _ = get_authorized_features()

    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"

    cols_to_read = ["SK_ID_CURR", "TARGET"] + feature_cols
    train_df = pd.read_parquet(train_path, columns=cols_to_read)
    val_df = pd.read_parquet(val_path, columns=cols_to_read)

    if use_sample:
        sample_ids = set(pd.read_csv(SAMPLE_IDS_PATH)["SK_ID_CURR"].tolist())
        train_df = train_df[train_df["SK_ID_CURR"].isin(sample_ids)].reset_index(drop=True)
        val_df = val_df[val_df["SK_ID_CURR"].isin(sample_ids)].reset_index(drop=True)

    # Verificar disyunción anti-leakage
    assert len(set(train_df["SK_ID_CURR"]).intersection(set(val_df["SK_ID_CURR"]))) == 0

    y_train = train_df["TARGET"].astype(np.int8)
    y_val = val_df["TARGET"].astype(np.int8)

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    # Asegurar que las columnas categóricas tengan exactamente el mismo CategoricalDtype ajustado en train
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        cats = X_train[c].cat.categories
        X_val[c] = pd.Categorical(X_val[c], categories=cats)

    # Convertir float64 residuales a float32 para consistencia y menor huella en RAM
    for c in num_cols:
        if X_train[c].dtype == np.float64:
            X_train[c] = X_train[c].astype(np.float32)
            X_val[c] = X_val[c].astype(np.float32)

    return X_train, y_train, X_val, y_val, num_cols, cat_cols


def find_optimal_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 1800.0,
    cost_fp: float = 150.0,
) -> dict[str, float]:
    """
    Busca en una rejilla de umbrales [0.05, 0.95] el punto de corte que minimiza
    el costo financiero esperado (FN = S/ 1,800, FP = S/ 150).
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    thresholds = np.linspace(0.05, 0.95, 181)

    best_thr = 0.50
    best_cost = float("inf")
    for thr in thresholds:
        y_pred = (y_prob_arr >= thr).astype(int)
        fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
        cost = float(fn * cost_fn + fp * cost_fp)
        if cost < best_cost:
            best_cost = cost
            best_thr = float(thr)

    return {
        "optimal_threshold": round(best_thr, 4),
        "min_total_cost": round(best_cost, 2),
        "min_cost_per_applicant": round(best_cost / len(y_true_arr), 4),
    }
