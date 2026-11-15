"""
Worker aislado para entrenar y validar XGBoost (`XGBClassifier`) en un proceso limpio
sin cargar `lib_lightgbm.dll` en el mismo espacio de memoria (evita el deadlock de OpenMP
entre `vcomp140.dll` de LightGBM y el runtime de XGBoost en Windows).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import classify_feature_family, load_train_val  # noqa: E402
from train_baseline import compute_full_metrics  # noqa: E402
from utils.custom_metrics import evaluate_binary_predictions, expected_cost  # noqa: E402
from data_loader import find_optimal_cost_threshold  # noqa: E402

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "04-modelado" / "salida_fase4"
TABLAS_DIR = OUT_DIR / "tablas"


def to_xgb_float32(df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Convierte el DataFrame a matriz `float32` contigua en memoria C, preservando `NaN`."""
    out = df.copy()
    for c in cat_cols:
        codes = out[c].cat.codes.to_numpy(dtype=np.float32)
        codes[codes < 0] = np.nan
        out[c] = codes
    return np.ascontiguousarray(out.to_numpy(dtype=np.float32))


def run_xgb_train(use_sample: bool = False) -> dict:
    print("[xgb_worker] Cargando datos en proceso aislado...", flush=True)
    X_train, y_train, X_val, y_val, num_cols, cat_cols = load_train_val(use_sample=use_sample)
    feat_names = X_train.columns.tolist()

    X_tr_arr = to_xgb_float32(X_train, cat_cols)
    X_va_arr = to_xgb_float32(X_val, cat_cols)
    y_tr_arr = y_train.to_numpy(dtype=np.int32)
    y_va_arr = y_val.to_numpy(dtype=np.int32)

    n_pos = int(y_tr_arr.sum())
    n_neg = int(len(y_tr_arr) - n_pos)
    scale_pos_weight = round(n_neg / n_pos, 4)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 450,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 80,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "early_stopping_rounds": 40,
        "random_state": 42,
        "n_jobs": 2,
    }
    print("[xgb_worker] Entrenando XGBClassifier (tree_method='hist', n_estimators=450, lr=0.05)...", flush=True)
    t0 = time.perf_counter()
    xgb_cand = xgb.XGBClassifier(**xgb_params)
    xgb_cand.fit(
        X_tr_arr,
        y_tr_arr,
        eval_set=[(X_va_arr, y_va_arr)],
        verbose=100,
    )
    t_xgb = time.perf_counter() - t0

    prob_xgb_train = xgb_cand.predict_proba(X_tr_arr)[:, 1]
    prob_xgb_val = xgb_cand.predict_proba(X_va_arr)[:, 1]

    m_xgb_tr = compute_full_metrics(y_tr_arr, prob_xgb_train)
    m_xgb_val = compute_full_metrics(y_va_arr, prob_xgb_val)
    print(
        f"[xgb_worker] Listo en {t_xgb:.1f}s | BestIter={xgb_cand.best_iteration} | "
        f"Train AUC={m_xgb_tr['roc_auc']:.4f} | Val AUC={m_xgb_val['roc_auc']:.4f} | "
        f"PR-AUC={m_xgb_val['pr_auc']:.4f} | KS={m_xgb_val['ks']:.4f}",
        flush=True,
    )

    joblib.dump(xgb_cand, MODELS_DIR / "modelo_candidato_xgb.pkl")

    xgb_imp = xgb_cand.feature_importances_
    df_imp_xgb = pd.DataFrame(
        {
            "variable": feat_names,
            "familia": [classify_feature_family(c) for c in feat_names],
            "gain_pct": np.round(100.0 * xgb_imp / max(np.sum(xgb_imp), 1e-9), 4),
        }
    ).sort_values("gain_pct", ascending=False).reset_index(drop=True)
    df_imp_xgb["ranking_gain"] = np.arange(1, len(df_imp_xgb) + 1)
    df_imp_xgb.to_csv(TABLAS_DIR / "feature_importance_xgb.csv", index=False)

    np.savez_compressed(
        OUT_DIR / "preds_xgb_only_val.npz",
        prob_xgb=prob_xgb_val.astype(np.float32),
    )

    res = {
        "hyperparameters": xgb_params,
        "best_iteration": int(xgb_cand.best_iteration),
        "train_time_sec": round(t_xgb, 3),
        "train": m_xgb_tr,
        "val": m_xgb_val,
        "generalization_gap_roc_auc": round(m_xgb_tr["roc_auc"] - m_xgb_val["roc_auc"], 5),
        "model_artifact": "models/modelo_candidato_xgb.pkl",
        "top_15_features": df_imp_xgb.head(15).to_dict(orient="records"),
    }
    with open(OUT_DIR / "metrics_xgb_only.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def run_xgb_cv(use_sample: bool = False, n_splits: int = 5) -> list[dict]:
    print("[xgb_worker] Ejecutando 5-Fold CV de XGBoost en proceso aislado...", flush=True)
    X_train, y_train, _, _, _, cat_cols = load_train_val(use_sample=use_sample)
    X_tr_all = to_xgb_float32(X_train, cat_cols)
    y_tr_all = y_train.to_numpy(dtype=np.int32)

    n_pos = int(y_tr_arr := y_tr_all.sum())
    n_neg = int(len(y_tr_all) - n_pos)
    scale_pos_weight = round(n_neg / n_pos, 4)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tr_all, y_tr_all), start=1):
        t0 = time.perf_counter()
        xgbm = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            n_estimators=260,
            learning_rate=0.06,
            max_depth=5,
            min_child_weight=80,
            scale_pos_weight=scale_pos_weight,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=2.0,
            reg_lambda=5.0,
            random_state=42 + fold_idx,
            n_jobs=-1,
        )
        xgbm.fit(
            X_tr_all[tr_idx],
            y_tr_all[tr_idx],
            verbose=False,
        )
        prob_xgb = xgbm.predict_proba(X_tr_all[va_idx])[:, 1]
        fit_time = time.perf_counter() - t0

        y_f_va = y_tr_all[va_idx]
        base_05 = evaluate_binary_predictions(y_f_va, prob_xgb, threshold=0.50)
        cost_05 = expected_cost(y_f_va, prob_xgb, threshold=0.50)
        opt_info = find_optimal_cost_threshold(y_f_va, prob_xgb)
        opt_thr = opt_info["optimal_threshold"]
        base_opt = evaluate_binary_predictions(y_f_va, prob_xgb, threshold=opt_thr)
        cost_opt = expected_cost(y_f_va, prob_xgb, threshold=opt_thr)

        row = {
            "modelo": "XGBoost (Candidato)",
            "fold": fold_idx,
            "n_val_fold": int(len(y_f_va)),
            "mora_pct_fold": round(100.0 * float(np.mean(y_f_va)), 4),
            "roc_auc": round(base_05["roc_auc"], 5),
            "pr_auc": round(base_05["pr_auc"], 5),
            "ks": round(base_05["ks"], 5),
            "mcc_050": round(base_05["mcc"], 5),
            "costo_solicitud_050": cost_05["cost_per_applicant"],
            "umbral_optimo_costo": opt_thr,
            "mcc_opt": round(base_opt["mcc"], 5),
            "costo_solicitud_opt": cost_opt["cost_per_applicant"],
            "tiempo_entrenamiento_seg": round(fit_time, 2),
        }
        rows.append(row)
        print(f"[xgb_worker] Fold {fold_idx}/{n_splits} -> XGB AUC={row['roc_auc']:.4f} ({fit_time:.1f}s)", flush=True)

    with open(OUT_DIR / "cv_xgb_only.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "cv"], default="train")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.mode == "train":
        run_xgb_train(use_sample=args.sample)
    else:
        run_xgb_cv(use_sample=args.sample)
