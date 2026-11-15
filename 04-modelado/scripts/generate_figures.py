"""
Generador de figuras técnicas de alta resolución (300 DPI) para la Fase 4: Modelado (CRISP-DM).
Produce en `04-modelado/figuras/`:
- `F01_curvas_roc_comparativa.png`: Curvas ROC en validación (Dummy, LogReg, LightGBM, XGBoost) + meta Charter (0.75).
- `F02_curvas_pr_comparativa.png`: Curvas Precision-Recall frente a la prevalencia de mora (8.07%).
- `F03_curva_ks_lgbm.png`: Estadístico Kolmogorov-Smirnov (separación acumulada TPR vs FPR) en LightGBM.
- `F04_boxplots_cv5fold.png`: Estabilidad de ROC-AUC y PR-AUC en los 5 folds de validación cruzada estratificada.
- `F05_importancia_variables_top25.png`: Top 25 variables predictoras por Gain (%) en LightGBM por familia relacional.
- `F06_distribucion_scores_clases.png`: Distribución de probabilidades predichas P(TARGET=1) y segmentación de zona gris.
- `diagrama_arquitectura.png`: Diagrama de arquitectura del prototipo de inferencia (FastAPI + Docker).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "04-modelado" / "salida_fase4"
TABLAS_DIR = OUT_DIR / "tablas"
FIG_DIR = ROOT / "04-modelado" / "figuras"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def plot_roc_and_pr_curves() -> None:
    b_npz = np.load(OUT_DIR / "preds_baseline_val.npz")
    g_npz = np.load(OUT_DIR / "preds_gbdt_val.npz")

    y_val = b_npz["y_val"]
    models_probs = [
        ("LightGBM (Candidato)", g_npz["prob_lgbm"], "#1b5e20", "-", 2.4),
        ("XGBoost (Candidato)", g_npz["prob_xgb"], "#0d47a1", "--", 2.0),
        ("Regresión Logística (Baseline)", b_npz["prob_logreg"], "#e65100", "-.", 2.0),
        ("Dummy (Most Frequent)", b_npz["prob_dummy"], "#757575", ":", 1.6),
    ]

    # F01: Curvas ROC Comparativas
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for label, prob, color, ls, lw in models_probs:
        fpr, tpr, _ = roc_curve(y_val, prob)
        auc_val = roc_auc_score(y_val, prob)
        ax.plot(fpr, tpr, label=f"{label} — ROC-AUC = {auc_val:.4f}", color=color, linestyle=ls, linewidth=lw)

    ax.axhline(0.75, color="#b71c1c", linestyle=":", alpha=0.55, linewidth=1.2)
    ax.text(
        0.52,
        0.16,
        "Meta Project Charter: ROC-AUC ≥ 0.7500\nUmbral salida proyecto: ROC-AUC < 0.7000",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#f57f17", alpha=0.9),
    )
    ax.set_title("Comparativa de Curvas ROC en Validación Hold-Out (n = 61,503)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR = 1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR = Sensibilidad / Recall)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F01_curvas_roc_comparativa.png")
    plt.close(fig)

    # F02: Curvas Precision-Recall Comparativas
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    base_rate = float(np.mean(y_val))
    for label, prob, color, ls, lw in models_probs[:-1]:
        prec, rec, _ = precision_recall_curve(y_val, prob)
        pr_auc = average_precision_score(y_val, prob)
        ax.plot(rec, prec, label=f"{label} — PR-AUC = {pr_auc:.4f}", color=color, linestyle=ls, linewidth=lw)

    ax.axhline(
        base_rate,
        color="#757575",
        linestyle=":",
        linewidth=1.8,
        label=f"Tasa base de mora (Dummy) — PR-AUC = {base_rate:.4f} (8.07%)",
    )
    ax.set_title("Curvas Precision-Recall en Validación Hold-Out (Desbalance 91.93% vs 8.07%)")
    ax.set_xlabel("Recall (Sensibilidad sobre Morosos Reales)")
    ax.set_ylabel("Precision (Valor Predictivo Positivo)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.75)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F02_curvas_pr_comparativa.png")
    plt.close(fig)

    # F03: Estadístico KS de LightGBM
    prob_lgbm = g_npz["prob_lgbm"]
    fpr, tpr, thresholds = roc_curve(y_val, prob_lgbm)
    # Filtrar umbrales infinitos
    valid_mask = np.isfinite(thresholds)
    fpr_v = fpr[valid_mask]
    tpr_v = tpr[valid_mask]
    thr_v = thresholds[valid_mask]
    diff = tpr_v - fpr_v
    ks_idx = int(np.argmax(diff))
    ks_val = float(diff[ks_idx])
    ks_thr = float(thr_v[ks_idx])

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(thr_v, tpr_v, label="TPR acumulado (Morosos: TARGET=1)", color="#c62828", linewidth=2.2)
    ax.plot(thr_v, fpr_v, label="FPR acumulado (Pagadores: TARGET=0)", color="#1565c0", linewidth=2.2)
    ax.vlines(
        ks_thr,
        fpr_v[ks_idx],
        tpr_v[ks_idx],
        color="#2e7d32",
        linestyle="--",
        linewidth=2.2,
        label=f"Máxima separación KS = {ks_val:.4f} (en score = {ks_thr:.3f})",
    )
    ax.scatter([ks_thr, ks_thr], [fpr_v[ks_idx], tpr_v[ks_idx]], color="#2e7d32", zorder=5, s=45)
    ax.set_title("Estadístico Kolmogorov-Smirnov (KS) — Modelo Candidato LightGBM (Validación)")
    ax.set_xlabel("Umbral de Probabilidad Predicha P(TARGET=1)")
    ax.set_ylabel("Proporción Acumulada")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F03_curva_ks_lgbm.png")
    plt.close(fig)


def plot_cv_boxplots() -> None:
    df_cv = pd.read_csv(OUT_DIR / "resultados_cv.csv")
    df_cv = df_cv[df_cv["modelo"] != "Dummy (Control)"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    modelos = ["Regresión Logística (Baseline)", "LightGBM (Candidato)", "XGBoost (Candidato)"]
    etiquetas = ["Regresión Logística\n(Baseline)", "LightGBM\n(Candidato)", "XGBoost\n(Candidato)"]
    colores = ["#fb8c00", "#2e7d32", "#1565c0"]

    # Subplot 1: ROC-AUC
    data_roc = [df_cv[df_cv["modelo"] == m]["roc_auc"].values for m in modelos]
    bp1 = axes[0].boxplot(data_roc, tick_labels=etiquetas, patch_artist=True, widths=0.45)
    for patch, c in zip(bp1["boxes"], colores):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for i, vals in enumerate(data_roc, start=1):
        axes[0].scatter(np.full_like(vals, i), vals, color="#212121", s=28, zorder=4)
        axes[0].text(
            i,
            float(np.mean(vals)) + 0.002,
            f"μ={np.mean(vals):.4f}\n±{np.std(vals, ddof=1):.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[0].axhline(0.75, color="#b71c1c", linestyle="--", linewidth=1.4, label="Meta Charter (0.75)")
    axes[0].set_title("ROC-AUC en Validación Cruzada (5-Fold)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(loc="lower right", fontsize=8.5)

    # Subplot 2: PR-AUC
    data_pr = [df_cv[df_cv["modelo"] == m]["pr_auc"].values for m in modelos]
    bp2 = axes[1].boxplot(data_pr, tick_labels=etiquetas, patch_artist=True, widths=0.45)
    for patch, c in zip(bp2["boxes"], colores):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for i, vals in enumerate(data_pr, start=1):
        axes[1].scatter(np.full_like(vals, i), vals, color="#212121", s=28, zorder=4)
        axes[1].text(
            i,
            float(np.mean(vals)) + 0.0025,
            f"μ={np.mean(vals):.4f}\n±{np.std(vals, ddof=1):.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1].set_title("PR-AUC en Validación Cruzada (5-Fold)")
    axes[1].set_ylabel("PR-AUC (Average Precision)")

    fig.suptitle("Estabilidad Estadística de los Modelos en 5-Fold Stratified CV (n = 246,008)", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F04_boxplots_cv5fold.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_top25() -> None:
    df_imp = pd.read_csv(TABLAS_DIR / "feature_importance_lgbm.csv").head(25).iloc[::-1]

    palette = {
        "Scores Externos (EXT_SOURCE)": "#1b5e20",
        "Pagos de Cuotas (installments)": "#0d47a1",
        "Buró Externo (bureau + balance)": "#00838f",
        "Solicitudes Previas y POS_CASH": "#6a1b9a",
        "Tarjeta de Crédito (credit_card)": "#ad1457",
        "Ratios Financieros y Montos": "#e65100",
        "Demográficas, Empleo y Estabilidad": "#4e342e",
        "Vivienda y Banderas de Ausencia": "#546e7a",
    }
    colors = [palette.get(fam, "#37474f") for fam in df_imp["familia"]]

    fig, ax = plt.subplots(figsize=(10.2, 7.6))
    bars = ax.barh(df_imp["variable"], df_imp["gain_pct"], color=colors, edgecolor="#263238", height=0.72)

    for bar, val in zip(bars, df_imp["gain_pct"]):
        ax.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    handles = [
        mpatches.Patch(color=col, label=fam)
        for fam, col in palette.items()
        if fam in df_imp["familia"].values
    ]
    ax.legend(handles=handles, title="Familia Relacional de la Variable", loc="lower right", frameon=True, fontsize=8.8)
    ax.set_title("Top 25 Variables Determinantes del Riesgo Crediticio — LightGBM (% Gain Total)")
    ax.set_xlabel("Participación en la Ganancia Total de Información del Modelo (% Gain)")
    ax.set_xlim(0, float(df_imp["gain_pct"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F05_importancia_variables_top25.png")
    plt.close(fig)


def plot_score_distribution() -> None:
    g_npz = np.load(OUT_DIR / "preds_gbdt_val.npz")
    y_val = g_npz["y_val"]
    prob_lgbm = g_npz["prob_lgbm"]

    with open(OUT_DIR / "metrics_gbdt.json", encoding="utf-8") as f:
        m_data = json.load(f)
    opt_thr = float(m_data["lightgbm_candidate"]["val"]["threshold_optimal_cost"]["threshold"])

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bins = np.linspace(0.0, 1.0, 65)
    ax.hist(
        prob_lgbm[y_val == 0],
        bins=bins,
        density=True,
        alpha=0.55,
        color="#1565c0",
        label="Pagadores Cumplidos (TARGET = 0, n = 56,538)",
    )
    ax.hist(
        prob_lgbm[y_val == 1],
        bins=bins,
        density=True,
        alpha=0.58,
        color="#c62828",
        label="Clientes en Mora (TARGET = 1, n = 4,965)",
    )

    # Zona gris preliminar e indicador de umbral óptimo por costo
    ax.axvspan(0.35, 0.60, color="#fff59d", alpha=0.40, label="Zona Gris Preliminar [0.35, 0.60] (Revisión Manual)")
    ax.axvline(opt_thr, color="#1b5e20", linestyle="--", linewidth=2.2, label=f"Umbral Óptimo por Costo (τ* = {opt_thr:.2f})")
    ax.axvline(0.50, color="#424242", linestyle=":", linewidth=1.6, label="Umbral Estándar (τ = 0.50)")

    ax.set_title("Distribución de Scores de Probabilidad (LightGBM con Pesos de Clase) y Segmentación")
    ax.set_xlabel("Score de Probabilidad Balanceado P(TARGET = 1 | X)")
    ax.set_ylabel("Densidad Empírica")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F06_distribucion_scores_clases.png")
    plt.close(fig)


def plot_architecture_diagram() -> None:
    """Genera el diagrama arquitectónico `diagrama_arquitectura.png` (Tarea F4.4)."""
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # Contenedor Docker principal
    docker_box = mpatches.FancyBboxPatch(
        (2.4, 0.5),
        6.5,
        4.3,
        boxstyle="round,pad=0.15",
        facecolor="#f1f8e9",
        edgecolor="#2e7d32",
        linewidth=2.0,
        linestyle="--",
    )
    ax.add_patch(docker_box)
    ax.text(
        5.65,
        4.55,
        "Contenedor Docker: home-credit-inference-service (Python 3.11-slim | Puerto 8000)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1b5e20",
    )

    def draw_box(x, y, w, h, title, subtitle, bg="#ffffff", edge="#1565c0"):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=bg,
            edgecolor=edge,
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#212121")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=8.0, color="#424242")

    # 1. Cliente externo
    draw_box(
        0.2,
        2.3,
        1.8,
        1.4,
        "Sistema Originación\n(Underwriting)",
        "POST /predict\nJSON: sk_id_curr\n+ 240 features",
        bg="#e3f2fd",
        edge="#0d47a1",
    )

    # 2. Capa FastAPI + Pydantic
    draw_box(
        2.7,
        2.8,
        2.6,
        1.3,
        "1. Capa REST FastAPI",
        "Uvicorn ASGI + Pydantic v2\nValidación anti-inyección\n/health | /model/info | /predict",
        bg="#ffffff",
        edge="#2e7d32",
    )

    # 3. Filtro de Gobernanza
    draw_box(
        5.9,
        2.8,
        2.7,
        1.3,
        "2. Filtro de Gobernanza",
        "Excluye CODE_GENDER, EDUC,\nFAMILY (Ley 29733 / SBS)\nAlinea 240 cols (float32/cat)",
        bg="#fff8e1",
        edge="#f57f17",
    )

    # 4. Motor LightGBM
    draw_box(
        5.9,
        0.9,
        2.7,
        1.3,
        "3. Motor LightGBM (RAM)",
        "modelo_candidato_lgbm.pkl\nManejo nativo NaN + FALTA_*\nInferencia < 20 ms",
        bg="#e8f5e9",
        edge="#1b5e20",
    )

    # 5. Regla de decisión y Zona Gris
    draw_box(
        2.7,
        0.9,
        2.6,
        1.3,
        "4. Segmentador Riesgo",
        "Aprobar (<0.35) | Zona Gris\n[0.35-0.60] | Rechazar (>0.60)\nLog sin PII (Ley 29733)",
        bg="#fce4ec",
        edge="#ad1457",
    )

    # 6. Volumen de modelos
    draw_box(
        9.2,
        2.8,
        2.1,
        1.3,
        "Volumen Host (:ro)",
        "./models/ -> /app/models\nmodelo_candidato_lgbm.pkl\nmetadata_inferencia.json",
        bg="#ede7f6",
        edge="#4a148c",
    )

    # 7. Respuesta JSON
    draw_box(
        9.2,
        0.9,
        2.1,
        1.3,
        "Respuesta Analítica",
        "probabilidad_impago\ndecision + zona_gris\ntop_factores + latencia_ms",
        bg="#e0f2f1",
        edge="#00695c",
    )

    # Flechas de flujo
    arrow_kw = dict(arrowstyle="->", color="#263238", linewidth=1.8, mutation_scale=14)
    ax.annotate("", xy=(2.7, 3.35), xytext=(2.0, 3.15), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.9, 3.45), xytext=(5.3, 3.45), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.25, 2.2), xytext=(7.25, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.3, 1.55), xytext=(5.9, 1.55), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.2, 1.35), xytext=(5.3, 1.25), arrowprops=dict(arrowstyle="->", color="#00695c", linewidth=1.8, mutation_scale=14))
    ax.annotate("", xy=(8.6, 1.75), xytext=(9.2, 3.25), arrowprops=dict(arrowstyle="->", color="#4a148c", linestyle="--", linewidth=1.6, mutation_scale=13))

    ax.set_title(
        "Arquitectura del Prototipo de Inferencia — Sistema Predictivo Home Credit (Fase 4 / Fase 6 CRISP-DM)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diagrama_arquitectura.png")
    plt.close(fig)


def run_generate_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("[generate_figures] Generando F01, F02 y F03 (Curvas ROC, PR y KS)...")
    plot_roc_and_pr_curves()
    print("[generate_figures] Generando F04 (Boxplots Validación Cruzada 5-Fold)...")
    plot_cv_boxplots()
    print("[generate_figures] Generando F05 (Importancia de Variables Top 25 LightGBM)...")
    plot_feature_importance_top25()
    print("[generate_figures] Generando F06 (Distribución de Scores y Zona Gris)...")
    plot_score_distribution()
    print("[generate_figures] Generando diagrama_arquitectura.png (Tarea F4.4)...")
    plot_architecture_diagram()
    print("[generate_figures] Todas las figuras fueron exportadas exitosamente en 04-modelado/figuras/")


if __name__ == "__main__":
    run_generate_figures()
