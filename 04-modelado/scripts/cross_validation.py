"""
4.3 Validación Cruzada Estratificada (5-Fold StratifiedKFold) y Comparativa Integral (`Cronograma` Fila 23).
- Evalúa sobre `train.parquet` (246,008 solicitudes) mediante `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  los 4 modelos del diseño experimental:
  1. `DummyClassifier (most_frequent)`
  2. `LogisticRegression (Pipeline anti-leakage, class_weight='balanced')`
  3. `LightGBM (Candidato regularizado, scale_pos_weight, NaN/category nativo)`
  4. `XGBoost (Candidato regularizado, hist + NaN nativo, scale_pos_weight)` en subproceso aislado (`run_xgb_worker.py`)
- Garantiza que todo preprocesamiento o early stopping se ajuste únicamente dentro de cada fold.
- Exporta `04-modelado/salida_fase4/resultados_cv.csv`, `04-modelado/salida_fase4/tablas/resumen_cv_modelos.csv`
  y la tabla consolidada `04-modelado/salida_fase4/comparativa_modelos.csv`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.custom_metrics import evaluate_binary_predictions, expected_cost  # noqa: E402
from data_loader import find_optimal_cost_threshold, load_train_val  # noqa: E402
from train_baseline import build_logreg_pipeline  # noqa: E402

OUT_DIR = ROOT / "04-modelado" / "salida_fase4"
TABLAS_DIR = OUT_DIR / "tablas"


def evaluate_fold_predictions(
    model_name: str,
    fold_idx: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fit_time_sec: float,
) -> dict:
    base_05 = evaluate_binary_predictions(y_true, y_prob, threshold=0.50)
    cost_05 = expected_cost(y_true, y_prob, threshold=0.50)
    opt_info = find_optimal_cost_threshold(y_true, y_prob)
    opt_thr = opt_info["optimal_threshold"]
    base_opt = evaluate_binary_predictions(y_true, y_prob, threshold=opt_thr)
    cost_opt = expected_cost(y_true, y_prob, threshold=opt_thr)

    return {
        "modelo": model_name,
        "fold": fold_idx,
        "n_val_fold": int(len(y_true)),
        "mora_pct_fold": round(100.0 * float(np.mean(y_true)), 4),
        "roc_auc": round(base_05["roc_auc"], 5),
        "pr_auc": round(base_05["pr_auc"], 5),
        "ks": round(base_05["ks"], 5),
        "mcc_050": round(base_05["mcc"], 5),
        "costo_solicitud_050": cost_05["cost_per_applicant"],
        "umbral_optimo_costo": opt_thr,
        "mcc_opt": round(base_opt["mcc"], 5),
        "costo_solicitud_opt": cost_opt["cost_per_applicant"],
        "tiempo_entrenamiento_seg": round(fit_time_sec, 2),
    }


def run_cross_validation(use_sample: bool = False, n_splits: int = 5) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[cross_validation] Cargando datos (use_sample={use_sample}, n_splits={n_splits})...", flush=True)
    X_train, y_train, X_val, y_val, num_cols, cat_cols = load_train_val(use_sample=use_sample)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = round(n_neg / n_pos, 4)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"[cross_validation] === Ejecutando Fold {fold_idx}/{n_splits} (Dummy, LogReg, LightGBM) ===", flush=True)
        X_f_tr = X_train.iloc[tr_idx].copy()
        y_f_tr = y_train.iloc[tr_idx].to_numpy()
        X_f_va = X_train.iloc[va_idx].copy()
        y_f_va = y_train.iloc[va_idx].to_numpy()

        # 1. Dummy
        t0 = time.perf_counter()
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_f_tr, y_f_tr)
        prob_dummy = dummy.predict_proba(X_f_va)[:, 1]
        cv_rows.append(
            evaluate_fold_predictions("Dummy (Control)", fold_idx, y_f_va, prob_dummy, time.perf_counter() - t0)
        )

        # 2. Logistic Regression Baseline
        t0 = time.perf_counter()
        lr_pipe = build_logreg_pipeline(num_cols, cat_cols, c_reg=0.1)
        lr_pipe.fit(X_f_tr, y_f_tr)
        prob_lr = lr_pipe.predict_proba(X_f_va)[:, 1]
        cv_rows.append(
            evaluate_fold_predictions(
                "Regresión Logística (Baseline)", fold_idx, y_f_va, prob_lr, time.perf_counter() - t0
            )
        )

        # 3. LightGBM Candidato
        t0 = time.perf_counter()
        lgbm = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=100,
            scale_pos_weight=scale_pos_weight,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=2.0,
            reg_lambda=5.0,
            cat_smooth=20.0,
            min_data_per_group=100,
            random_state=42 + fold_idx,
            n_jobs=4,
            verbose=-1,
        )
        lgbm.fit(
            X_f_tr,
            y_f_tr,
            eval_set=[(X_f_va, y_f_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=40, first_metric_only=True, verbose=False)],
        )
        prob_lgbm = lgbm.predict_proba(X_f_va)[:, 1]
        cv_rows.append(
            evaluate_fold_predictions("LightGBM (Candidato)", fold_idx, y_f_va, prob_lgbm, time.perf_counter() - t0)
        )

        print(
            f"[cross_validation] Fold {fold_idx} -> "
            f"LogReg AUC={cv_rows[-2]['roc_auc']:.4f} | "
            f"LGBM AUC={cv_rows[-1]['roc_auc']:.4f}",
            flush=True,
        )

    # 4. Ejecutar 5-Fold CV de XGBoost en proceso aislado
    print("[cross_validation] === Ejecutando 5-Fold CV de XGBoost en subproceso aislado ===", flush=True)
    worker_script = Path(__file__).resolve().parent / "run_xgb_worker.py"
    cmd = [sys.executable, str(worker_script), "--mode", "cv"]
    if use_sample:
        cmd.append("--sample")
    subprocess.run(cmd, check=True)

    with open(OUT_DIR / "cv_xgb_only.json", encoding="utf-8") as f:
        xgb_cv_rows = json.load(f)
    cv_rows.extend(xgb_cv_rows)

    df_cv = pd.DataFrame(cv_rows).sort_values(["fold", "modelo"]).reset_index(drop=True)
    df_cv.to_csv(OUT_DIR / "resultados_cv.csv", index=False)
    df_cv.to_csv(TABLAS_DIR / "resultados_cv.csv", index=False)

    # Resumen agregado media ± std por modelo en CV
    order_models = [
        "Dummy (Control)",
        "Regresión Logística (Baseline)",
        "LightGBM (Candidato)",
        "XGBoost (Candidato)",
    ]
    resumen_rows = []
    for model_name in order_models:
        grp = df_cv[df_cv["modelo"] == model_name]
        resumen_rows.append(
            {
                "modelo": model_name,
                "n_folds": len(grp),
                "cv_roc_auc_mean": round(float(grp["roc_auc"].mean()), 5),
                "cv_roc_auc_std": round(float(grp["roc_auc"].std(ddof=1)), 5),
                "cv_pr_auc_mean": round(float(grp["pr_auc"].mean()), 5),
                "cv_pr_auc_std": round(float(grp["pr_auc"].std(ddof=1)), 5),
                "cv_ks_mean": round(float(grp["ks"].mean()), 5),
                "cv_ks_std": round(float(grp["ks"].std(ddof=1)), 5),
                "cv_mcc_050_mean": round(float(grp["mcc_050"].mean()), 5),
                "cv_mcc_opt_mean": round(float(grp["mcc_opt"].mean()), 5),
                "cv_costo_opt_mean": round(float(grp["costo_solicitud_opt"].mean()), 2),
                "cv_tiempo_mean_seg": round(float(grp["tiempo_entrenamiento_seg"].mean()), 2),
                "cumple_meta_charter_075": bool(grp["roc_auc"].mean() >= 0.75),
            }
        )
    df_resumen_cv = pd.DataFrame(resumen_rows)
    df_resumen_cv.to_csv(TABLAS_DIR / "resumen_cv_modelos.csv", index=False)

    # Si ya existen metrics_baseline.json y metrics_gbdt.json, consolidar tabla maestra comparativa_modelos.csv
    baseline_json = OUT_DIR / "metrics_baseline.json"
    gbdt_json = OUT_DIR / "metrics_gbdt.json"
    if baseline_json.exists() and gbdt_json.exists():
        with open(baseline_json, encoding="utf-8") as f:
            b_data = json.load(f)
        with open(gbdt_json, encoding="utf-8") as f:
            g_data = json.load(f)

        model_map = {
            "Dummy (Control)": b_data["dummy_control"],
            "Regresión Logística (Baseline)": b_data["logistic_regression_baseline"],
            "LightGBM (Candidato)": g_data["lightgbm_candidate"],
            "XGBoost (Candidato)": g_data["xgboost_candidate"],
        }

        comp_rows = []
        for row_cv in resumen_rows:
            m_name = row_cv["modelo"]
            m_info = model_map[m_name]
            tr = m_info["train"]
            va = m_info["val"]
            comp_rows.append(
                {
                    "modelo": m_name,
                    "cv5_roc_auc_mean": row_cv["cv_roc_auc_mean"],
                    "cv5_roc_auc_std": row_cv["cv_roc_auc_std"],
                    "train_roc_auc": tr["roc_auc"],
                    "val_roc_auc": va["roc_auc"],
                    "gap_train_val_auc": round(tr["roc_auc"] - va["roc_auc"], 5),
                    "val_pr_auc": va["pr_auc"],
                    "val_ks": va["ks"],
                    "val_accuracy_050": va["threshold_050"]["accuracy"],
                    "val_recall_050": va["threshold_050"]["recall"],
                    "val_precision_050": va["threshold_050"]["precision"],
                    "val_mcc_050": va["threshold_050"]["mcc"],
                    "val_costo_solicitud_050_pen": va["threshold_050"]["cost_per_applicant_pen"],
                    "val_umbral_opt_costo": va["threshold_optimal_cost"]["threshold"],
                    "val_recall_opt": va["threshold_optimal_cost"]["recall"],
                    "val_precision_opt": va["threshold_optimal_cost"]["precision"],
                    "val_mcc_opt": va["threshold_optimal_cost"]["mcc"],
                    "val_costo_solicitud_opt_pen": va["threshold_optimal_cost"]["cost_per_applicant_pen"],
                    "tiempo_entrenamiento_full_seg": m_info["train_time_sec"],
                    "supera_criterio_charter_075": bool(va["roc_auc"] >= 0.75),
                }
            )
        df_comp = pd.DataFrame(comp_rows)
        df_comp.to_csv(OUT_DIR / "comparativa_modelos.csv", index=False)
        df_comp.to_csv(TABLAS_DIR / "comparativa_modelos.csv", index=False)

    print("[cross_validation] Validación cruzada finalizada y guardada en resultados_cv.csv", flush=True)
    return df_cv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Ejecutar sobre muestra 10%")
    args = parser.parse_args()
    run_cross_validation(use_sample=args.sample)
