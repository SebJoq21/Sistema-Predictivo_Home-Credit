"""
Script para sincronizar las celdas de fase4_modelado.ipynb con Prioridad 2 y 3.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "04-modelado" / "fase4_modelado.ipynb"


def patch_notebook() -> None:
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        # Portada
        if cell["cell_type"] == "markdown" and "Fase 4: Modelado (CRISP-DM)" in "".join(cell["source"]):
            cell["source"] = [
                "# Fase 4: Modelado (CRISP-DM) — Notebook de Experimentación y Validación\n\n",
                "**Proyecto:** Sistema Predictivo de Riesgo Crediticio (*Home Credit Default Risk*) · **Grupo 12 — UNMSM**  \n",
                "**Rúbricas cubiertas:**  \n",
                "- **4.1 Selección de técnicas de modelado:** Progresión ordenada de menor a mayor complejidad (Niveles 0 a 4: `DummyClassifier`, `LogisticRegression`, `RandomForest` ref., `XGBoost`, `LightGBM`), justificada con **casos reales bancarios/Fintech** y literatura en revistas **Q1, Q2 y Q3** (Scopus/WoS).\n",
                "- **4.2 Plan de pruebas:** Matriz formal de casos de prueba (`TC-01` a `TC-07`), aislamiento muestral anti-leakage y validación cruzada estratificada de 5 pliegues (`5-Fold StratifiedKFold`).\n",
                "- **4.3 Construcción y ajuste de parámetros:** Estudio de sensibilidad, regularización estructural e importancia de variables por ganancia (*Gain*).\n",
                "- **4.4 Evaluación técnica, calibración e inferencia:** Comparativa integral, Curva de Calibración / Brier Score, pruebas de significancia pareadas ($t=23.80, p < 0.001$) y prototipo `FastAPI`."
            ]

        # Plan de pruebas
        if cell["cell_type"] == "markdown" and "1. Verificación del Conjunto de Datos" in "".join(cell["source"]):
            cell["source"] = [
                "## 1. Verificación de Datos, Gobernanza y Plan de Pruebas (Rúbricas 4.1 y 4.2)\n",
                "Se certifican los insumos de Fase 3 y la matriz de casos de prueba (`TC-01` a `TC-07`): **240 variables predictoras autorizadas** (`incluir_modelo == True`), exclusión estricta de variables protegidas (`CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`) y particiones disyuntas ($0$ clientes compartidos)."
            ]

        # CV 5-Fold y significancia
        if cell["cell_type"] == "markdown" and "4. Validación Cruzada" in "".join(cell["source"]):
            cell["source"] = [
                "## 4. Validación Cruzada Estratificada de 5 Pliegues, Significancia Estadística y Comparativa Maestra (`Tarea F4.3`)\n",
                "Resultados por pliegue (`resultados_cv.csv`), resumen estadístico ($\\mu \\pm \\sigma$), pruebas de significancia pareadas (`pruebas_significancia_cv.csv`: Paired t-test $t=23.80, p = 1.85 \\times 10^{-5}$) y tabla comparativa final en `val.parquet`."
            ]

        # Curvas de evaluación y calibración
        if cell["cell_type"] == "markdown" and "5. Curvas de Evaluación Técnica" in "".join(cell["source"]):
            cell["source"] = [
                "## 5. Curvas de Evaluación Técnica y Calibración (ROC, Precision-Recall, KS, Boxplots CV y Brier Score)"
            ]

        # Código que carga imágenes
        if cell["cell_type"] == "code" and any("F04_boxplots_cv5fold.png" in s for s in cell["source"]):
            cell["source"] = [
                "for fig_name in [\n",
                "    'F01_curvas_roc_comparativa.png',\n",
                "    'F02_curvas_pr_comparativa.png',\n",
                "    'F03_curva_ks_lgbm.png',\n",
                "    'F04_boxplots_cv5fold.png',\n",
                "    'F07_curvas_calibracion_brier.png',\n",
                "]:\n",
                "    display(Image(filename=str(FIG_DIR / fig_name), width=780))"
            ]

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[patch_notebook] {NB_PATH.name} actualizado exitosamente con Prioridad 2 y 3.", flush=True)


if __name__ == "__main__":
    patch_notebook()
