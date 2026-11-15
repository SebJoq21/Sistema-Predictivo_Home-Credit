"""
4.1 Entrenamiento del Modelo Baseline — Regresión Logística + Control Dummy (`Cronograma` Fila 21).
- Demuestra cuantitativamente por qué el Accuracy (91.93%) es engañoso bajo desbalance severo
  mediante un clasificador Dummy (`most_frequent`).
- Construye y entrena un Pipeline de Regresión Logística con pesos de clase (`class_weight='balanced'`),
  regularización L2 (`C=0.1`) y preprocesamiento sin fuga de datos (`SimpleImputer(median)` + `StandardScaler`
  en numéricas y `OrdinalEncoder` + `StandardScaler` en categóricas, ajustados exclusivamente sobre `train`).
- Guarda `models/modelo_baseline_logreg.pkl`, `04-modelado/salida_fase4/metrics_baseline.json` y
  `04-modelado/salida_fase4/tablas/coeficientes_logreg_top25.csv`.
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.custom_metrics import evaluate_binary_predictions, expected_cost  # noqa: E402
from data_loader import classify_feature_family, find_optimal_cost_threshold, load_train_val  # noqa: E402

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "04-modelado" / "salida_fase4"
TABLAS_DIR = OUT_DIR / "tablas"


def build_logreg_pipeline(num_cols: list[str], cat_cols: list[str], c_reg: float = 0.1) -> Pipeline:
    """Construye el Pipeline anti-leakage para Regresión Logística."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        C=c_reg,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=500,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def compute_full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    """Calcula métricas obligatorias del Charter, complementarias y costo financiero."""
    base = evaluate_binary_predictions(y_true, y_prob, threshold=threshold)
    cost = expected_cost(y_true, y_prob, threshold=threshold)
    opt_cost = find_optimal_cost_threshold(y_true, y_prob)
    opt_thr = opt_cost["optimal_threshold"]
    base_opt = evaluate_binary_predictions(y_true, y_prob, threshold=opt_thr)
    cost_opt = expected_cost(y_true, y_prob, threshold=opt_thr)

    y_pred_05 = (np.asarray(y_prob) >= threshold).astype(int)
    y_pred_opt = (np.asarray(y_prob) >= opt_thr).astype(int)

    return {
        "roc_auc": round(base["roc_auc"], 5),
        "pr_auc": round(base["pr_auc"], 5),
        "ks": round(base["ks"], 5),
        "threshold_050": {
            "threshold": 0.50,
            "accuracy": round(float(accuracy_score(y_true, y_pred_05)), 5),
            "precision": round(float(precision_score(y_true, y_pred_05, zero_division=0)), 5),
            "recall": round(float(recall_score(y_true, y_pred_05, zero_division=0)), 5),
            "f1": round(float(f1_score(y_true, y_pred_05, zero_division=0)), 5),
            "mcc": round(base["mcc"], 5),
            "tn": cost["tn"],
            "fp": cost["fp"],
            "fn": cost["fn"],
            "tp": cost["tp"],
            "total_cost_pen": round(cost["total_cost"], 2),
            "cost_per_applicant_pen": cost["cost_per_applicant"],
        },
        "threshold_optimal_cost": {
            "threshold": opt_thr,
            "accuracy": round(float(accuracy_score(y_true, y_pred_opt)), 5),
            "precision": round(float(precision_score(y_true, y_pred_opt, zero_division=0)), 5),
            "recall": round(float(recall_score(y_true, y_pred_opt, zero_division=0)), 5),
            "f1": round(float(f1_score(y_true, y_pred_opt, zero_division=0)), 5),
            "mcc": round(base_opt["mcc"], 5),
            "tn": cost_opt["tn"],
            "fp": cost_opt["fp"],
            "fn": cost_opt["fn"],
            "tp": cost_opt["tp"],
            "total_cost_pen": round(cost_opt["total_cost"], 2),
            "cost_per_applicant_pen": cost_opt["cost_per_applicant"],
        },
    }


def run_train_baseline(use_sample: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train_baseline] Cargando datos (use_sample={use_sample})...")
    X_train, y_train, X_val, y_val, num_cols, cat_cols = load_train_val(use_sample=use_sample)
    print(
        f"[train_baseline] Train: {X_train.shape[0]:,} filas x {X_train.shape[1]} features "
        f"(num={len(num_cols)}, cat={len(cat_cols)}) | Val: {X_val.shape[0]:,} filas"
    )

    # 1. Modelo Dummy (Control Estadístico)
    t0 = time.perf_counter()
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    t_dummy = time.perf_counter() - t0
    prob_dummy_train = dummy.predict_proba(X_train)[:, 1]
    prob_dummy_val = dummy.predict_proba(X_val)[:, 1]

    metrics_dummy_train = compute_full_metrics(y_train.to_numpy(), prob_dummy_train)
    metrics_dummy_val = compute_full_metrics(y_val.to_numpy(), prob_dummy_val)

    # 2. Modelo Baseline: Regresión Logística con Pipeline Anti-Leakage
    print("[train_baseline] Entrenando Pipeline de Regresión Logística (class_weight='balanced', C=0.1)...")
    t0 = time.perf_counter()
    logreg_pipe = build_logreg_pipeline(num_cols, cat_cols, c_reg=0.1)
    logreg_pipe.fit(X_train, y_train)
    t_logreg = time.perf_counter() - t0

    prob_lr_train = logreg_pipe.predict_proba(X_train)[:, 1]
    prob_lr_val = logreg_pipe.predict_proba(X_val)[:, 1]

    metrics_lr_train = compute_full_metrics(y_train.to_numpy(), prob_lr_train)
    metrics_lr_val = compute_full_metrics(y_val.to_numpy(), prob_lr_val)

    # Guardar modelo serializado
    model_path = MODELS_DIR / "modelo_baseline_logreg.pkl"
    joblib.dump(logreg_pipe, model_path)

    # Extraer coeficientes estandarizados
    ordered_features = num_cols + cat_cols
    coefs = logreg_pipe.named_steps["classifier"].coef_[0]
    df_coefs = pd.DataFrame(
        {
            "variable": ordered_features,
            "familia": [classify_feature_family(c) for c in ordered_features],
            "coeficiente_estandarizado": np.round(coefs, 6),
            "abs_coeficiente": np.round(np.abs(coefs), 6),
            "odds_ratio_1sd": np.round(np.exp(coefs), 6),
            "direccion_riesgo": np.where(coefs > 0, "Aumenta mora (+)", "Reduce mora (-)"),
        }
    ).sort_values("abs_coeficiente", ascending=False)
    df_coefs.head(25).to_csv(TABLAS_DIR / "coeficientes_logreg_top25.csv", index=False)

    # Guardar probabilidades de validación para curvas comparativas
    np.savez_compressed(
        OUT_DIR / "preds_baseline_val.npz",
        y_val=y_val.to_numpy(),
        prob_dummy=prob_dummy_val.astype(np.float32),
        prob_logreg=prob_lr_val.astype(np.float32),
    )

    summary = {
        "dataset": {
            "use_sample": use_sample,
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_numerical": len(num_cols),
            "n_categorical": len(cat_cols),
            "train_mora_pct": round(100.0 * float(y_train.mean()), 4),
            "val_mora_pct": round(100.0 * float(y_val.mean()), 4),
        },
        "dummy_control": {
            "strategy": "most_frequent",
            "train_time_sec": round(t_dummy, 3),
            "train": metrics_dummy_train,
            "val": metrics_dummy_val,
        },
        "logistic_regression_baseline": {
            "hyperparameters": {
                "imputer_num": "SimpleImputer(strategy='median')",
                "scaler_num": "StandardScaler()",
                "encoder_cat": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-2)",
                "penalty": "l2",
                "C": 0.1,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": 42,
            },
            "train_time_sec": round(t_logreg, 3),
            "n_iter_actual": int(logreg_pipe.named_steps["classifier"].n_iter_[0]),
            "train": metrics_lr_train,
            "val": metrics_lr_val,
            "generalization_gap_roc_auc": round(
                metrics_lr_train["roc_auc"] - metrics_lr_val["roc_auc"], 5
            ),
            "model_artifact": "models/modelo_baseline_logreg.pkl",
        },
    }

    with open(OUT_DIR / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"[train_baseline] Dummy Val -> Acc={metrics_dummy_val['threshold_050']['accuracy']:.4f}, "
        f"ROC-AUC={metrics_dummy_val['roc_auc']:.4f}, MCC={metrics_dummy_val['threshold_050']['mcc']:.4f}"
    )
    print(
        f"[train_baseline] LogReg Train -> ROC-AUC={metrics_lr_train['roc_auc']:.4f}, "
        f"PR-AUC={metrics_lr_train['pr_auc']:.4f}, KS={metrics_lr_train['ks']:.4f}"
    )
    print(
        f"[train_baseline] LogReg Val   -> ROC-AUC={metrics_lr_val['roc_auc']:.4f}, "
        f"PR-AUC={metrics_lr_val['pr_auc']:.4f}, KS={metrics_lr_val['ks']:.4f}, "
        f"MCC(0.5)={metrics_lr_val['threshold_050']['mcc']:.4f}, "
        f"MCC(opt)={metrics_lr_val['threshold_optimal_cost']['mcc']:.4f} "
        f"({t_logreg:.2f}s)"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Ejecutar sobre muestra 10%")
    args = parser.parse_args()
    run_train_baseline(use_sample=args.sample)
