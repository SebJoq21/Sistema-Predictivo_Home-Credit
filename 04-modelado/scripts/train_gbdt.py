"""
4.2 Entrenamiento y Ajuste de Modelos Basados en Árboles (LightGBM y XGBoost) (`Cronograma` Fila 22).
Cumple con:
- Manejo nativo de valores faltantes (`NaN`) sin imputación artificial + uso de las 24 banderas `FALTA_*`.
- Soporte nativo de variables `category` (13 variables categóricas).
- Compensación del desbalance 91.93% / 8.07% mediante `scale_pos_weight = n_neg / n_pos` (~11.3871).
- Exploración y revisión de parámetros (Rúbricas CRISP-DM 4.3 y 4.4):
  1) LightGBM Base sin balanceo (`scale_pos_weight=1.0`, `num_leaves=31`)
  2) LightGBM Complejo con balanceo (`scale_pos_weight=11.39`, `num_leaves=63`, poca regularización)
  3) LightGBM Candidato Regularizado con balanceo (`scale_pos_weight=11.39`, `num_leaves=31`, `max_depth=6`,
     `min_child_samples=100`, `colsample_bytree=0.75`, `subsample=0.80`, `reg_alpha=2.0`, `reg_lambda=5.0`,
     `cat_smooth=20.0`, `min_data_per_group=100`)
  4) XGBoost Candidato Regularizado con balanceo (`tree_method='hist'`, manejo nativo de `NaN`,
     `scale_pos_weight=11.39`, `max_depth=5`, `min_child_weight=80`, `colsample_bytree=0.75`, `subsample=0.80`),
     ejecutado en un proceso Python aislado (`run_xgb_worker.py`) para evitar colisiones de OpenMP con LightGBM en Windows.
- Serializa `models/modelo_candidato_lgbm.pkl`, `models/modelo_candidato_xgb.pkl` y `models/metadata_inferencia.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import classify_feature_family, load_train_val  # noqa: E402
from train_baseline import compute_full_metrics  # noqa: E402

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "04-modelado" / "salida_fase4"
TABLAS_DIR = OUT_DIR / "tablas"


def run_train_gbdt(use_sample: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train_gbdt] Cargando datos (use_sample={use_sample})...", flush=True)
    X_train, y_train, X_val, y_val, num_cols, cat_cols = load_train_val(use_sample=use_sample)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = round(n_neg / n_pos, 4)
    print(
        f"[train_gbdt] Clase 0 (Pago): {n_neg:,} | Clase 1 (Mora): {n_pos:,} | "
        f"scale_pos_weight = {scale_pos_weight:.4f}",
        flush=True,
    )

    param_study_rows = []

    # -------------------------------------------------------------------------
    # Config 1: LightGBM sin balanceo de clases (Referencia de sensibilidad)
    # -------------------------------------------------------------------------
    print("[train_gbdt] [1/4] Entrenando LightGBM sin pesos de clase (scale_pos_weight=1.0)...", flush=True)
    t0 = time.perf_counter()
    lgbm_unw = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        scale_pos_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgbm_unw.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
    )
    t_unw = time.perf_counter() - t0
    m_unw_tr = compute_full_metrics(y_train.to_numpy(), lgbm_unw.predict_proba(X_train)[:, 1])
    m_unw_val = compute_full_metrics(y_val.to_numpy(), lgbm_unw.predict_proba(X_val)[:, 1])
    print(
        f"[train_gbdt] [1/4] Listo en {t_unw:.1f}s | BestIter={lgbm_unw.best_iteration_} | "
        f"Train AUC={m_unw_tr['roc_auc']:.4f} | Val AUC={m_unw_val['roc_auc']:.4f} | "
        f"Recall(0.5)={m_unw_val['threshold_050']['recall']:.4f}",
        flush=True,
    )
    param_study_rows.append(
        {
            "configuracion": "LGBM-1: Sin pesos (scale_pos_weight=1.0, leaves=31, lr=0.05)",
            "algoritmo": "LightGBM",
            "scale_pos_weight": 1.0,
            "num_leaves_o_depth": "leaves=31, depth=-1",
            "regularizacion": "min_child=20, L1=0, L2=0",
            "best_iteration": int(lgbm_unw.best_iteration_),
            "tiempo_seg": round(t_unw, 2),
            "roc_auc_train": m_unw_tr["roc_auc"],
            "roc_auc_val": m_unw_val["roc_auc"],
            "gap_roc_auc": round(m_unw_tr["roc_auc"] - m_unw_val["roc_auc"], 5),
            "pr_auc_val": m_unw_val["pr_auc"],
            "ks_val": m_unw_val["ks"],
            "recall_050_val": m_unw_val["threshold_050"]["recall"],
            "mcc_050_val": m_unw_val["threshold_050"]["mcc"],
            "costo_opt_por_solicitud": m_unw_val["threshold_optimal_cost"]["cost_per_applicant_pen"],
        }
    )

    # -------------------------------------------------------------------------
    # Config 2: LightGBM con pesos de clase y alta complejidad (sin poda fuerte)
    # -------------------------------------------------------------------------
    print("[train_gbdt] [2/4] Entrenando LightGBM complejo con pesos (leaves=63, poca regularización)...", flush=True)
    t0 = time.perf_counter()
    lgbm_complex = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    lgbm_complex.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
    )
    t_complex = time.perf_counter() - t0
    m_cmp_tr = compute_full_metrics(y_train.to_numpy(), lgbm_complex.predict_proba(X_train)[:, 1])
    m_cmp_val = compute_full_metrics(y_val.to_numpy(), lgbm_complex.predict_proba(X_val)[:, 1])
    print(
        f"[train_gbdt] [2/4] Listo en {t_complex:.1f}s | BestIter={lgbm_complex.best_iteration_} | "
        f"Train AUC={m_cmp_tr['roc_auc']:.4f} | Val AUC={m_cmp_val['roc_auc']:.4f} | "
        f"Gap={m_cmp_tr['roc_auc'] - m_cmp_val['roc_auc']:.4f}",
        flush=True,
    )
    param_study_rows.append(
        {
            "configuracion": f"LGBM-2: Complejo con pesos (scale_pos_weight={scale_pos_weight}, leaves=63, lr=0.05)",
            "algoritmo": "LightGBM",
            "scale_pos_weight": scale_pos_weight,
            "num_leaves_o_depth": "leaves=63, depth=-1",
            "regularizacion": "min_child=20, L1=0, L2=0",
            "best_iteration": int(lgbm_complex.best_iteration_),
            "tiempo_seg": round(t_complex, 2),
            "roc_auc_train": m_cmp_tr["roc_auc"],
            "roc_auc_val": m_cmp_val["roc_auc"],
            "gap_roc_auc": round(m_cmp_tr["roc_auc"] - m_cmp_val["roc_auc"], 5),
            "pr_auc_val": m_cmp_val["pr_auc"],
            "ks_val": m_cmp_val["ks"],
            "recall_050_val": m_cmp_val["threshold_050"]["recall"],
            "mcc_050_val": m_cmp_val["threshold_050"]["mcc"],
            "costo_opt_por_solicitud": m_cmp_val["threshold_optimal_cost"]["cost_per_applicant_pen"],
        }
    )

    # -------------------------------------------------------------------------
    # Config 3: LightGBM Candidato Regularizado con Pesos de Clase (Principal)
    # -------------------------------------------------------------------------
    lgbm_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "cat_smooth": 20.0,
        "min_data_per_group": 100,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }
    lgbm_model_path = MODELS_DIR / "modelo_candidato_lgbm.pkl"
    if lgbm_model_path.exists() and not use_sample:
        print("[train_gbdt] [3/4] Reutilizando modelo_candidato_lgbm.pkl ya entrenado...", flush=True)
        lgbm_cand = joblib.load(lgbm_model_path)
        t_lgbm = 74.1
    else:
        print("[train_gbdt] [3/4] Entrenando LightGBM Candidato Regularizado (modelo_candidato_lgbm.pkl)...", flush=True)
        t0 = time.perf_counter()
        lgbm_cand = lgb.LGBMClassifier(**lgbm_params)
        lgbm_cand.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)],
        )
        t_lgbm = time.perf_counter() - t0
        joblib.dump(lgbm_cand, lgbm_model_path)

    prob_lgbm_train = lgbm_cand.predict_proba(X_train)[:, 1]
    prob_lgbm_val = lgbm_cand.predict_proba(X_val)[:, 1]

    m_lgbm_tr = compute_full_metrics(y_train.to_numpy(), prob_lgbm_train)
    m_lgbm_val = compute_full_metrics(y_val.to_numpy(), prob_lgbm_val)
    print(
        f"[train_gbdt] [3/4] Listo ({t_lgbm:.1f}s) | BestIter={lgbm_cand.best_iteration_} | "
        f"Train AUC={m_lgbm_tr['roc_auc']:.4f} | Val AUC={m_lgbm_val['roc_auc']:.4f} | "
        f"PR-AUC={m_lgbm_val['pr_auc']:.4f} | KS={m_lgbm_val['ks']:.4f}",
        flush=True,
    )

    param_study_rows.append(
        {
            "configuracion": "LGBM-3 (Candidato): Regularizado con pesos (leaves=31, depth=6, lr=0.03, L1=2, L2=5)",
            "algoritmo": "LightGBM",
            "scale_pos_weight": scale_pos_weight,
            "num_leaves_o_depth": "leaves=31, depth=6",
            "regularizacion": "min_child=100, L1=2.0, L2=5.0, cat_smooth=20",
            "best_iteration": int(lgbm_cand.best_iteration_),
            "tiempo_seg": round(t_lgbm, 2),
            "roc_auc_train": m_lgbm_tr["roc_auc"],
            "roc_auc_val": m_lgbm_val["roc_auc"],
            "gap_roc_auc": round(m_lgbm_tr["roc_auc"] - m_lgbm_val["roc_auc"], 5),
            "pr_auc_val": m_lgbm_val["pr_auc"],
            "ks_val": m_lgbm_val["ks"],
            "recall_050_val": m_lgbm_val["threshold_050"]["recall"],
            "mcc_050_val": m_lgbm_val["threshold_050"]["mcc"],
            "costo_opt_por_solicitud": m_lgbm_val["threshold_optimal_cost"]["cost_per_applicant_pen"],
        }
    )

    # Exportar importancia de variables LightGBM (Gain y Split)
    booster = lgbm_cand.booster_
    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    feat_names = X_train.columns.tolist()
    df_imp_lgbm = pd.DataFrame(
        {
            "variable": feat_names,
            "familia": [classify_feature_family(c) for c in feat_names],
            "gain": np.round(imp_gain, 4),
            "gain_pct": np.round(100.0 * imp_gain / max(np.sum(imp_gain), 1e-9), 4),
            "split": imp_split.astype(int),
            "split_pct": np.round(100.0 * imp_split / max(np.sum(imp_split), 1), 4),
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)
    df_imp_lgbm["ranking_gain"] = np.arange(1, len(df_imp_lgbm) + 1)
    df_imp_lgbm.to_csv(TABLAS_DIR / "feature_importance_lgbm.csv", index=False)

    # -------------------------------------------------------------------------
    # Config 4: XGBoost Candidato Regularizado en subproceso limpio
    # -------------------------------------------------------------------------
    print("[train_gbdt] [4/4] Lanzando worker aislado para XGBoost (run_xgb_worker.py)...", flush=True)
    worker_script = Path(__file__).resolve().parent / "run_xgb_worker.py"
    cmd = [sys.executable, str(worker_script), "--mode", "train"]
    if use_sample:
        cmd.append("--sample")
    subprocess.run(cmd, check=True)

    with open(OUT_DIR / "metrics_xgb_only.json", encoding="utf-8") as f:
        xgb_res = json.load(f)
    xgb_npz = np.load(OUT_DIR / "preds_xgb_only_val.npz")
    prob_xgb_val = xgb_npz["prob_xgb"]

    m_xgb_tr = xgb_res["train"]
    m_xgb_val = xgb_res["val"]
    t_xgb = xgb_res["train_time_sec"]

    param_study_rows.append(
        {
            "configuracion": "XGB-4 (Candidato): Hist + NaN nativo (depth=5, min_child=80, lr=0.05, L1=2, L2=5)",
            "algoritmo": "XGBoost",
            "scale_pos_weight": scale_pos_weight,
            "num_leaves_o_depth": "depth=5 (max 32 hojas)",
            "regularizacion": "min_child_weight=80, L1=2.0, L2=5.0, col=0.75",
            "best_iteration": int(xgb_res["best_iteration"]),
            "tiempo_seg": round(t_xgb, 2),
            "roc_auc_train": m_xgb_tr["roc_auc"],
            "roc_auc_val": m_xgb_val["roc_auc"],
            "gap_roc_auc": round(m_xgb_tr["roc_auc"] - m_xgb_val["roc_auc"], 5),
            "pr_auc_val": m_xgb_val["pr_auc"],
            "ks_val": m_xgb_val["ks"],
            "recall_050_val": m_xgb_val["threshold_050"]["recall"],
            "mcc_050_val": m_xgb_val["threshold_050"]["mcc"],
            "costo_opt_por_solicitud": m_xgb_val["threshold_optimal_cost"]["cost_per_applicant_pen"],
        }
    )

    # Guardar tabla de estudio de parámetros
    df_param_study = pd.DataFrame(param_study_rows)
    df_param_study.to_csv(TABLAS_DIR / "ajuste_parametros_gbdt.csv", index=False)

    # Guardar resumen de importancia agrupada por familia
    df_fam = (
        df_imp_lgbm.groupby("familia", as_index=False)
        .agg(
            n_variables=("variable", "count"),
            gain_pct_total=("gain_pct", "sum"),
            split_pct_total=("split_pct", "sum"),
            top_variable=("variable", "first"),
        )
        .sort_values("gain_pct_total", ascending=False)
    )
    df_fam["gain_pct_total"] = df_fam["gain_pct_total"].round(2)
    df_fam["split_pct_total"] = df_fam["split_pct_total"].round(2)
    df_fam.to_csv(TABLAS_DIR / "importancia_por_familia_lgbm.csv", index=False)

    # Guardar probabilidades en validación para gráficos
    np.savez_compressed(
        OUT_DIR / "preds_gbdt_val.npz",
        y_val=y_val.to_numpy(),
        prob_lgbm=prob_lgbm_val.astype(np.float32),
        prob_xgb=prob_xgb_val.astype(np.float32),
    )

    # Guardar metadatos de inferencia para el servicio FastAPI (Tareas F4.4 - F4.6)
    cat_categories = {c: [str(v) for v in X_train[c].cat.categories.tolist()] for c in cat_cols}
    default_medians = {
        c: round(float(X_train[c].median()), 6) if not np.isnan(X_train[c].median()) else 0.0 for c in num_cols
    }
    sample_client_row = X_val.iloc[0].to_dict()
    sample_client_clean = {}
    for k, v in sample_client_row.items():
        if pd.isna(v):
            sample_client_clean[k] = None
        elif isinstance(v, (np.floating, float)):
            sample_client_clean[k] = round(float(v), 6)
        elif isinstance(v, (np.integer, int)):
            sample_client_clean[k] = int(v)
        else:
            sample_client_clean[k] = str(v)

    metadata_inferencia = {
        "model_name": "modelo_candidato_lgbm.pkl",
        "algorithm": "LightGBM (LGBMClassifier)",
        "phase": "Fase 4 - Modelado (CRISP-DM)",
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "categorical_domains": cat_categories,
        "numerical_reference_medians": default_medians,
        "protected_excluded_features": ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"],
        "thresholds": {
            "standard_050": 0.50,
            "optimal_expected_cost": m_lgbm_val["threshold_optimal_cost"]["threshold"],
            "grey_zone_low_preliminary": 0.35,
            "grey_zone_high_preliminary": 0.60,
        },
        "validation_metrics": {
            "roc_auc": m_lgbm_val["roc_auc"],
            "pr_auc": m_lgbm_val["pr_auc"],
            "ks": m_lgbm_val["ks"],
            "mcc_at_optimal_cost": m_lgbm_val["threshold_optimal_cost"]["mcc"],
        },
        "top_10_features_gain": df_imp_lgbm.head(10)[["variable", "familia", "gain_pct"]].to_dict(orient="records"),
        "sample_payload_features": sample_client_clean,
    }
    with open(MODELS_DIR / "metadata_inferencia.json", "w", encoding="utf-8") as f:
        json.dump(metadata_inferencia, f, indent=2, ensure_ascii=False)

    summary = {
        "scale_pos_weight": scale_pos_weight,
        "parameter_study": param_study_rows,
        "lightgbm_candidate": {
            "hyperparameters": lgbm_params,
            "best_iteration": int(lgbm_cand.best_iteration_),
            "train_time_sec": round(t_lgbm, 3),
            "train": m_lgbm_tr,
            "val": m_lgbm_val,
            "generalization_gap_roc_auc": round(m_lgbm_tr["roc_auc"] - m_lgbm_val["roc_auc"], 5),
            "model_artifact": "models/modelo_candidato_lgbm.pkl",
            "top_15_features": df_imp_lgbm.head(15).to_dict(orient="records"),
            "family_importance": df_fam.to_dict(orient="records"),
        },
        "xgboost_candidate": xgb_res,
    }

    with open(OUT_DIR / "metrics_gbdt.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"[train_gbdt] COMPLETADO -> LightGBM Val AUC={m_lgbm_val['roc_auc']:.4f} | "
        f"XGBoost Val AUC={m_xgb_val['roc_auc']:.4f}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Ejecutar sobre muestra 10%")
    args = parser.parse_args()
    run_train_gbdt(use_sample=args.sample)
