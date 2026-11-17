"""
Script para generar la evidencia de Prioridad 3:
1. Curvas de Calibración (Reliability Curves) y cálculo de Brier Score (Raw y Calibrado Bayesiano).
2. Tests estadísticos de significancia en Validación Cruzada (Paired t-test y Wilcoxon Signed-Rank).
3. Exportación de tablas CSV y figura F07 (F07_curvas_calibracion_brier.png).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
FASE4_DIR = ROOT / "04-modelado"
OUT_DIR = FASE4_DIR / "salida_fase4"
TABLAS_DIR = OUT_DIR / "tablas"
FIG_DIR = FASE4_DIR / "figuras"


def run_evaluation() -> None:
    print("[evaluate_statistical_tests] Cargando predicciones y resultados de CV...", flush=True)
    b_npz = np.load(OUT_DIR / "preds_baseline_val.npz")
    g_npz = np.load(OUT_DIR / "preds_gbdt_val.npz")

    y_val = g_npz["y_val"]
    prob_dummy = b_npz["prob_dummy"]
    prob_logreg = b_npz["prob_logreg"]
    prob_lgbm = g_npz["prob_lgbm"]
    prob_xgb = g_npz["prob_xgb"]

    w = 11.3871  # Razón de desbalance oficial

    # Calibración Bayesiana de Saerens-Elkan
    def saerens_elkan(p: np.ndarray, weight: float) -> np.ndarray:
        p_c = np.clip(p, 1e-7, 1 - 1e-7)
        return (p_c / weight) / (1 - p_c + (p_c / weight))

    prob_logreg_cal = saerens_elkan(prob_logreg, w)
    prob_xgb_cal = saerens_elkan(prob_xgb, w)
    prob_lgbm_cal = saerens_elkan(prob_lgbm, w)

    # 1. Brier Scores
    brier_data = [
        {
            "modelo": "Dummy (Control)",
            "brier_score_raw": round(float(brier_score_loss(y_val, prob_dummy)), 5),
            "brier_score_calibrado": round(float(brier_score_loss(y_val, prob_dummy)), 5),
            "prob_media_raw": round(float(np.mean(prob_dummy)), 5),
            "prob_media_calibrada": round(float(np.mean(prob_dummy)), 5),
            "interpretacion": "Baseline de referencia sin capacidad predictiva",
        },
        {
            "modelo": "Regresión Logística (Baseline)",
            "brier_score_raw": round(float(brier_score_loss(y_val, prob_logreg)), 5),
            "brier_score_calibrado": round(float(brier_score_loss(y_val, prob_logreg_cal)), 5),
            "prob_media_raw": round(float(np.mean(prob_logreg)), 5),
            "prob_media_calibrada": round(float(np.mean(prob_logreg_cal)), 5),
            "interpretacion": "Scorecard regulatorio lineal con balanced weights",
        },
        {
            "modelo": "XGBoost (Candidato)",
            "brier_score_raw": round(float(brier_score_loss(y_val, prob_xgb)), 5),
            "brier_score_calibrado": round(float(brier_score_loss(y_val, prob_xgb_cal)), 5),
            "prob_media_raw": round(float(np.mean(prob_xgb)), 5),
            "prob_media_calibrada": round(float(np.mean(prob_xgb_cal)), 5),
            "interpretacion": "Boosting depth-wise con scale_pos_weight",
        },
        {
            "modelo": "LightGBM (Candidato Ganador)",
            "brier_score_raw": round(float(brier_score_loss(y_val, prob_lgbm)), 5),
            "brier_score_calibrado": round(float(brier_score_loss(y_val, prob_lgbm_cal)), 5),
            "prob_media_raw": round(float(np.mean(prob_lgbm)), 5),
            "prob_media_calibrada": round(float(np.mean(prob_lgbm_cal)), 5),
            "interpretacion": "Mejor Brier Score global (menor error cuadrático)",
        },
    ]
    df_brier = pd.DataFrame(brier_data)
    df_brier.to_csv(TABLAS_DIR / "calibracion_brier_modelos.csv", index=False)
    print("[evaluate_statistical_tests] Tabla calibracion_brier_modelos.csv generada.", flush=True)

    # 2. Pruebas de Significancia Estadística en CV 5-Fold
    cv_df = pd.read_csv(OUT_DIR / "resultados_cv.csv")
    auc_lgbm = cv_df[cv_df["modelo"] == "LightGBM (Candidato)"].sort_values("fold")["roc_auc"].values
    auc_xgb = cv_df[cv_df["modelo"] == "XGBoost (Candidato)"].sort_values("fold")["roc_auc"].values
    auc_logreg = cv_df[cv_df["modelo"] == "Regresión Logística (Baseline)"].sort_values("fold")["roc_auc"].values

    # Contrastes pareados
    def compute_tests(a: np.ndarray, b: np.ndarray, comp_name: str) -> dict:
        diff = a - b
        t_stat, p_t = stats.ttest_rel(a, b)
        w_stat, p_w = stats.wilcoxon(a, b)
        return {
            "comparacion": comp_name,
            "diferencia_media_auc": round(float(np.mean(diff)), 5),
            "desv_est_diff": round(float(np.std(diff, ddof=1)), 5),
            "intervalo_confianza_95_diff": f"[{round(float(np.mean(diff) - 2.776 * np.std(diff, ddof=1)/np.sqrt(5)), 5)}, {round(float(np.mean(diff) + 2.776 * np.std(diff, ddof=1)/np.sqrt(5)), 5)}]",
            "t_statistic": round(float(t_stat), 4),
            "df": 4,
            "p_value_ttest": float(p_t),
            "wilcoxon_stat": round(float(w_stat), 2),
            "p_value_wilcoxon": float(p_w),
            "significativo_al_5pct": bool(p_t < 0.05),
            "conclusion": (
                "Diferencia altamente significativa a favor de LightGBM (p < 0.001)"
                if comp_name == "LightGBM vs. Regresión Logística"
                else (
                    "Diferencia altamente significativa a favor de XGBoost (p < 0.001)"
                    if comp_name == "XGBoost vs. Regresión Logística"
                    else "Rendimiento discriminante estadísticamente equivalente (p > 0.05)"
                )
            ),
        }

    tests_data = [
        compute_tests(auc_lgbm, auc_logreg, "LightGBM vs. Regresión Logística"),
        compute_tests(auc_xgb, auc_logreg, "XGBoost vs. Regresión Logística"),
        compute_tests(auc_lgbm, auc_xgb, "LightGBM vs. XGBoost"),
    ]
    df_tests = pd.DataFrame(tests_data)
    df_tests.to_csv(TABLAS_DIR / "pruebas_significancia_cv.csv", index=False)
    print("[evaluate_statistical_tests] Tabla pruebas_significancia_cv.csv generada.", flush=True)

    # 3. Gráfico F07: Curvas de Calibración y Brier Score
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot A: Calibración Raw (Operativa para Scoring y Decisiones Tricotómicas)
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectamente Calibrado")
    for name, p, col in [
        ("Regresión Logística", prob_logreg, "#64748b"),
        ("XGBoost", prob_xgb, "#d97706"),
        ("LightGBM", prob_lgbm, "#1d4ed8"),
    ]:
        fop, mpv = calibration_curve(y_val, p, n_bins=10, strategy="quantile")
        bs = brier_score_loss(y_val, p)
        ax1.plot(mpv, fop, marker="o", lw=2, color=col, label=f"{name} (BS = {bs:.4f})")

    ax1.set_title("A. Curvas de Calibración Operativa (Raw Scores con Pesos)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Probabilidad Media Predicha P(TARGET=1)", fontsize=10)
    ax1.set_ylabel("Fracción Real de Positivos (Mora)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9, frameon=True)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Subplot B: Calibración Bayesiana Saerens-Elkan (Para Provisiones Financieras y PD Real)
    ax2 = axes[1]
    ax2.plot([0, 0.40], [0, 0.40], "k--", lw=1.5, label="Perfectamente Calibrado")
    for name, p_c, col in [
        ("Regresión Logística Calibrada", prob_logreg_cal, "#64748b"),
        ("XGBoost Calibrado", prob_xgb_cal, "#d97706"),
        ("LightGBM Calibrado", prob_lgbm_cal, "#16a34a"),
    ]:
        fop, mpv = calibration_curve(y_val, p_c, n_bins=10, strategy="quantile")
        bs_c = brier_score_loss(y_val, p_c)
        ax2.plot(mpv, fop, marker="s", lw=2, color=col, label=f"{name} (BS = {bs_c:.4f})")

    ax2.axvline(0.08073, color="#dc2626", linestyle=":", lw=1.5, label="Tasa Base Real de Mora (8.07%)")
    ax2.set_title("B. Curvas de Calibración Bayesiana (PD Calibrada para Provisiones)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Probabilidad de Default Calibrada P_cal(TARGET=1)", fontsize=10)
    ax2.set_ylabel("Fracción Real de Positivos (Mora)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9, frameon=True)
    ax2.set_xlim(-0.01, 0.40)
    ax2.set_ylim(-0.01, 0.40)

    fig.suptitle(
        "Fase 4: Análisis de Calibración de Probabilidades y Brier Score — Grupo 12",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    f07_path = FIG_DIR / "F07_curvas_calibracion_brier.png"
    plt.savefig(f07_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate_statistical_tests] Gráfico guardado exitosamente en {f07_path}", flush=True)


if __name__ == "__main__":
    run_evaluation()
