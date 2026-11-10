"""
Métricas personalizadas de evaluación y costo esperado para clasificación desbalanceada
(CRISP-DM Fases 3-5 · Home Credit Default Risk).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula el estadístico Kolmogorov-Smirnov (KS) máximo entre TPR y FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = 1800.0,
    cost_fp: float = 150.0,
) -> dict[str, float]:
    """
    Calcula el costo financiero esperado según los supuestos del Business Case:
    - Falso Negativo (moroso aprobado): S/ 1,800 (S/ 3,000 * 60% LGD)
    - Falso Positivo (buen cliente rechazado): S/ 150 (margen perdido)
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = float(fn * cost_fn + fp * cost_fp)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": total_cost,
        "cost_per_applicant": round(total_cost / len(y_true), 4),
    }


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula las métricas obligatorias del Charter: ROC-AUC, PR-AUC, KS y MCC."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
