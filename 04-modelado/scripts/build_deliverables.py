"""
Script integrador de cierre de la Fase 4: Modelado (CRISP-DM):
1. Verifica funcionalmente el servicio de inferencia FastAPI (`services/inference-service/main.py`)
   sobre perfiles reales de `val.parquet` (aprobación automática, zona gris, rechazo automático y lote)
   y guarda la evidencia en `04-modelado/salida_fase4/prueba_servicio_inferencia.json`.
2. Construye y ejecuta el Jupyter Notebook `04-modelado/fase4_modelado.ipynb` con todos los resultados,
   tablas comparativas y gráficos integrados.
3. Renderiza `04-modelado/04_informe_modelado.md` a `04-modelado/04-MODELADO.pdf` con el estilo académico
   corporativo del proyecto (idéntico a `03-PREPARACIÓN DE DATOS.pdf`).
4. Limpia el archivo placeholder vacío `04-modelado/texto`.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
FASE4_DIR = ROOT / "04-modelado"
OUT_DIR = FASE4_DIR / "salida_fase4"
FIG_DIR = FASE4_DIR / "figuras"


def test_inference_service() -> dict:
    """Prueba extremo a extremo de `services/inference-service/main.py` con TestClient."""
    print("[build_deliverables] Probando servicio FastAPI (services/inference-service/main.py)...", flush=True)
    from fastapi.testclient import TestClient
    import numpy as np
    import pandas as pd

    service_path = ROOT / "services" / "inference-service" / "main.py"
    spec = importlib.util.spec_from_file_location("inference_main", service_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inference_main"] = mod
    spec.loader.exec_module(mod)

    client = TestClient(mod.app)

    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Error en /health: {r_health.text}"
    r_info = client.get("/model/info")
    assert r_info.status_code == 200, f"Error en /model/info: {r_info.text}"

    # Seleccionar 3 perfiles reales de val.parquet (bajo riesgo, zona gris, alto riesgo)
    g_npz = np.load(OUT_DIR / "preds_gbdt_val.npz")
    probs = g_npz["prob_lgbm"]
    val_df = pd.read_parquet(ROOT / "data" / "processed" / "val.parquet")

    idx_low = int(np.where(probs < 0.20)[0][0])
    idx_grey = int(np.where((probs >= 0.42) & (probs <= 0.52))[0][0])
    idx_high = int(np.where(probs > 0.75)[0][0])

    def row_to_payload(idx: int) -> dict:
        row = val_df.iloc[idx]
        sk_id = int(row["SK_ID_CURR"])
        feats = {}
        for k, v in row.items():
            if k in ("SK_ID_CURR", "TARGET"):
                continue
            if pd.isna(v):
                feats[k] = None
            elif hasattr(v, "item"):
                feats[k] = v.item()
            else:
                feats[k] = v
        return {"sk_id_curr": sk_id, "features": feats}

    single_results = []
    for label, idx in [
        ("perfil_bajo_riesgo", idx_low),
        ("perfil_zona_gris", idx_grey),
        ("perfil_alto_riesgo", idx_high),
    ]:
        payload = row_to_payload(idx)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, f"Error en /predict ({label}): {resp.text}"
        data = resp.json()
        data["perfil_prueba"] = label
        data["target_real_val"] = int(val_df.iloc[idx]["TARGET"])
        single_results.append(data)

    batch_payload = {"applications": [row_to_payload(i) for i in range(10)]}
    r_batch = client.post("/predict/batch", json=batch_payload)
    assert r_batch.status_code == 200, f"Error en /predict/batch: {r_batch.text}"

    # Estadísticas de cobertura de zona gris sobre las 61,503 solicitudes de val.parquet
    pct_auto_approve = round(100.0 * float(np.mean(probs < 0.35)), 2)
    pct_grey_zone = round(100.0 * float(np.mean((probs >= 0.35) & (probs <= 0.60))), 2)
    pct_auto_reject = round(100.0 * float(np.mean(probs > 0.60)), 2)
    pct_total_auto = round(pct_auto_approve + pct_auto_reject, 2)

    evidence = {
        "health_endpoint": r_health.json(),
        "model_info_summary": {
            "model_name": r_info.json()["model_name"],
            "n_features": r_info.json()["n_features"],
            "protected_excluded_features": r_info.json()["protected_excluded_features"],
            "thresholds": r_info.json()["thresholds"],
        },
        "single_predictions_verified": single_results,
        "batch_prediction_verified": r_batch.json(),
        "val_segmentation_coverage": {
            "total_val_applicants": int(len(probs)),
            "aprobar_automatico_pct": pct_auto_approve,
            "zona_gris_manual_pct": pct_grey_zone,
            "rechazar_automatico_pct": pct_auto_reject,
            "automatizacion_total_pct": pct_total_auto,
        },
    }
    with open(OUT_DIR / "prueba_servicio_inferencia.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(
        f"[build_deliverables] Servicio FastAPI verificado OK | Automatización en val = {pct_total_auto}% "
        f"(Zona gris = {pct_grey_zone}%)",
        flush=True,
    )
    return evidence


def build_and_execute_notebook() -> None:
    """Construye y ejecuta `04-modelado/fase4_modelado.ipynb`."""
    print("[build_deliverables] Construyendo y ejecutando 04-modelado/fase4_modelado.ipynb...", flush=True)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = [
        new_markdown_cell(
            "# Fase 4: Modelado (CRISP-DM) — Notebook de Experimentación y Validación\n\n"
            "**Proyecto:** Sistema Predictivo de Riesgo Crediticio (*Home Credit Default Risk*) · **Grupo 12 — UNMSM**  \n"
            "**Rúbricas cubiertas:**  \n"
            "- **4.1 Selección de técnicas de modelado:** Progresión ordenada de menor a mayor complejidad (Niveles 0 a 4: `DummyClassifier`, `LogisticRegression`, `RandomForest` ref., `XGBoost`, `LightGBM`), justificada con **casos reales de la industria bancaria/Fintech** y literatura científica en revistas indexadas **Q1, Q2 y Q3** (Scopus/WoS).\n"
            "- **4.2 Diseño de pruebas:** Validación Cruzada Estratificada de 5 pliegues (`5-Fold StratifiedKFold`) sobre `train.parquet` ($246,008$ solicitudes) y evaluación *hold-out* en `val.parquet` ($61,503$ solicitudes).\n"
            "- **4.3 Construcción y ajuste de parámetros:** Estudio de sensibilidad, regularización e importancia de variables por ganancia (*Gain*).\n"
            "- **4.4 Evaluación técnica e inferencia:** Comparativa de métricas (`ROC-AUC`, `PR-AUC`, `KS`, `MCC`, Costo Esperado) y verificación del microservicio `FastAPI`."
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path('..').resolve()\n"
            "OUT_DIR = Path('salida_fase4')\n"
            "TABLAS_DIR = OUT_DIR / 'tablas'\n"
            "FIG_DIR = Path('figuras')\n"
            "pd.set_option('display.max_columns', 30)\n"
            "pd.set_option('display.width', 140)\n"
            "print('Directorio de artefactos:', OUT_DIR.resolve())"
        ),
        new_markdown_cell(
            "## 1. Verificación del Conjunto de Datos y Gobernanza de Variables (Rúbricas 4.1 y 4.2)\n"
            "Se cargan los metadatos del entrenamiento para confirmar el uso exclusivo de las **240 variables predictoras autorizadas** (`incluir_modelo == True`), excluyendo `CODE_GENDER`, `NAME_EDUCATION_TYPE` y `NAME_FAMILY_STATUS`."
        ),
        new_code_cell(
            "with open(OUT_DIR / 'metrics_baseline.json', encoding='utf-8') as f:\n"
            "    baseline_meta = json.load(f)\n"
            "with open(ROOT / 'models' / 'metadata_inferencia.json', encoding='utf-8') as f:\n"
            "    inf_meta = json.load(f)\n\n"
            "print('Resumen del Dataset (Train / Val):')\n"
            "display(pd.DataFrame([baseline_meta['dataset']]))\n"
            "print('Variables protegidas excluidas del modelo (Ley N.° 29733 / SBS):', inf_meta['protected_excluded_features'])"
        ),
        new_markdown_cell(
            "## 2. Selección de Técnicas de Modelado (Rúbrica 4.1), Baseline Lineal y Control Dummy\n"
            "En estricto cumplimiento de la Rúbrica 4.1 de CRISP-DM y las pautas académicas:\n\n"
            "### 2.1 Escalera de Complejidad Algorítmica (De Menor a Mayor Complejidad):\n"
            "1. **Nivel 0 (Nula - Control Trivial):** `DummyClassifier` (predice `TARGET=0`; $Acc=91.93\\%$, $AUC=0.5000$, $MCC=0.0000$, $Recall=0.0000$).\n"
            "2. **Nivel 1 (Baja - Paramétrico Lineal):** `LogisticRegression` con `Pipeline` anti-leakage (`SimpleImputer(median)` + `StandardScaler` + penalización $L_2$, $C=0.1$, `class_weight='balanced'`). Baseline regulatorio bancario.\n"
            "3. **Nivel 2 (Media - Ensamble por Bagging):** Árboles de Decisión / Random Forest (reducción de varianza por promediado ortogonal; costo $O(B \\cdot N \\log N \\cdot d)$).\n"
            "4. **Nivel 3 (Alta - Boosting Depth-Wise):** `XGBoost` (`tree_method='hist'`, profundidad $\\le 5$, regularización $L_1/L_2$, aprendizaje de dirección para `NaN`).\n"
            "5. **Nivel 4 (Alta/Avanzada - Boosting Leaf-Wise con Histogramas):** `LightGBM` (partición por hoja de mayor ganancia, cuantización en $255$ bins, soporte nativo de `category` y enrutamiento óptimo de `NaN`).\n\n"
            "### 2.2 Justificación en Casos Reales de la Industria:\n"
            "- **Banca Regulada (BBVA, Santander, BCP — Basilea II/III / SBS):** Scorecards logísticos tradicionales por explicabilidad directa y aditividad.\n"
            "- **Fintechs y Neobancos (Home Credit, Nubank, LendingClub):** Adopción de GBDT (LightGBM/XGBoost) para créditos no tradicionales, logrando mejoras de $+2.5\\text{ pp}$ a $+3.0\\text{ pp}$ en ROC-AUC y reducciones del $15\\%$ al $25\\%$ en mora bruta.\n"
            "- **Burós Internacionales (FICO, Experian):** Arquitectura tricotómica (Aprobación automática, Zona Gris y Rechazo automático).\n\n"
            "### 2.3 Sustentación en Papers Indexados (Scopus / WoS):\n"
            "- **Paper Q1 (EJOR):** Lessmann et al. (2015) — Benchmark de 41 clasificadores en credit scoring que demuestra la superioridad estadística de los ensambles de árboles sobre la regresión logística ($p < 0.01$).\n"
            "- **Paper Q1 (ESWA):** Xia et al. (2017) — Robustez de árboles potenciados por gradiente con regularización ante ruido y alta colinealidad.\n"
            "- **Paper Q2 (Comp. Econ.):** Petropoulos et al. (2020) — Captura de interacciones no lineales complejas en carteras retail bancarias europeas.\n"
            "- **Paper Q3 (IJACSA):** Al-Auda & Al-Mayyahi (2022) — Técnicas para desbalance severo de clases (1:11.4) y uso de pesos de clase."
        ),
        new_code_cell(
            "df_coefs = pd.read_csv(TABLAS_DIR / 'coeficientes_logreg_top25.csv')\n"
            "print('Top 15 Coeficientes Estandarizados — Regresión Logística (Baseline):')\n"
            "display(df_coefs.head(15))"
        ),
        new_markdown_cell(
            "## 3. Estudio de Sensibilidad y Ajuste de Parámetros en GBDT (`Tarea F4.2` — Rúbricas 4.3 y 4.4)\n"
            "Se comparan las 4 configuraciones experimentales de árboles potenciados por gradiente (`LightGBM` sin pesos, `LightGBM` complejo con pesos, `LightGBM` candidato regularizado y `XGBoost` candidato regularizado)."
        ),
        new_code_cell(
            "df_params = pd.read_csv(TABLAS_DIR / 'ajuste_parametros_gbdt.csv')\n"
            "display(df_params)"
        ),
        new_markdown_cell(
            "## 4. Validación Cruzada Estratificada de 5 Pliegues y Comparativa Maestra (`Tarea F4.3`)\n"
            "Resultados por pliegue (`resultados_cv.csv`), resumen estadístico ($\\mu \\pm \\sigma$) y tabla comparativa final en `val.parquet`."
        ),
        new_code_cell(
            "df_cv_resumen = pd.read_csv(TABLAS_DIR / 'resumen_cv_modelos.csv')\n"
            "df_comp = pd.read_csv(OUT_DIR / 'comparativa_modelos.csv')\n"
            "print('=== Resumen 5-Fold Stratified Cross-Validation (train.parquet, n = 246,008) ===')\n"
            "display(df_cv_resumen)\n"
            "print('=== Tabla Comparativa Maestra (CV 5-Fold + Hold-Out val.parquet, n = 61,503) ===')\n"
            "display(df_comp)"
        ),
        new_markdown_cell(
            "## 5. Curvas de Evaluación Técnica (ROC, Precision-Recall, KS y Boxplots CV)"
        ),
        new_code_cell(
            "for fig_name in [\n"
            "    'F01_curvas_roc_comparativa.png',\n"
            "    'F02_curvas_pr_comparativa.png',\n"
            "    'F03_curva_ks_lgbm.png',\n"
            "    'F04_boxplots_cv5fold.png',\n"
            "]:\n"
            "    display(Image(filename=str(FIG_DIR / fig_name), width=780))"
        ),
        new_markdown_cell(
            "## 6. Patrones Predictivos Extraídos e Importancia de Variables por Familia Relacional\n"
            "Se analiza la contribución porcentual a la ganancia de información (*Gain*) de las 8 familias de variables y el Top-25 de atributos más determinantes del modelo ganador `LightGBM`."
        ),
        new_code_cell(
            "df_fam = pd.read_csv(TABLAS_DIR / 'importancia_por_familia_lgbm.csv')\n"
            "df_imp_lgbm = pd.read_csv(TABLAS_DIR / 'feature_importance_lgbm.csv')\n"
            "print('=== Importancia Agrupada por Familia Relacional (LightGBM) ===')\n"
            "display(df_fam)\n"
            "print('=== Top 20 Variables por Ganancia de Información (% Gain) ===')\n"
            "display(df_imp_lgbm.head(20))\n"
            "display(Image(filename=str(FIG_DIR / 'F05_importancia_variables_top25.png'), width=820))\n"
            "display(Image(filename=str(FIG_DIR / 'F06_distribucion_scores_clases.png'), width=780))"
        ),
        new_markdown_cell(
            "## 7. Verificación del Prototipo del Servicio de Inferencia (`FastAPI` + `Docker` — Tareas F4.4 a F4.6)\n"
            "Se inspecciona el diagrama de arquitectura y la prueba funcional de los endpoints `/health`, `/model/info`, `/predict` y `/predict/batch` sobre perfiles reales de validación."
        ),
        new_code_cell(
            "display(Image(filename=str(FIG_DIR / 'diagrama_arquitectura.png'), width=860))\n"
            "with open(OUT_DIR / 'prueba_servicio_inferencia.json', encoding='utf-8') as f:\n"
            "    prueba_api = json.load(f)\n"
            "print('Estado /health:', prueba_api['health_endpoint'])\n"
            "print('Cobertura de Segmentación en Validación (Aprobación / Zona Gris / Rechazo):')\n"
            "display(pd.DataFrame([prueba_api['val_segmentation_coverage']]))\n"
            "print('Predicciones individuales verificadas en /predict:')\n"
            "display(pd.DataFrame(prueba_api['single_predictions_verified'])[['perfil_prueba', 'sk_id_curr', 'target_real_val', 'probabilidad_impago', 'decision', 'requiere_revision_manual', 'latencia_ms']])"
        ),
    ]

    nb.cells = cells
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(FASE4_DIR)}})
    client.execute()

    nb_path = FASE4_DIR / "fase4_modelado.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[build_deliverables] Notebook ejecutado y guardado en {nb_path}", flush=True)


def _format_inline_md(text: str) -> str:
    """Convierte sintaxis inline de Markdown (negritas, cursivas, código, fórmulas simples) a XML de ReportLab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Reemplazar símbolos LaTeX frecuentes por equivalentes Unicode limpios
    latex_replacements = {
        r"\approx": "≈",
        r"\ge": "≥",
        r"\le": "≤",
        r"\times": "×",
        r"\pm": "±",
        r"\rightarrow": "→",
        r"\Delta": "Δ",
        r"\mu": "μ",
        r"\sigma": "σ",
        r"\tau^*": "τ*",
        r"\tau": "τ",
        r"\beta_0": "β₀",
        r"\beta": "β",
        r"\boldsymbol{\beta}": "β",
        r"\alpha": "α",
        r"\lambda": "λ",
        r"\chi^2": "χ²",
        r"\%": "%",
        r"\text{ROC-AUC}_{\text{train}}": "ROC-AUC_train",
        r"\text{ROC-AUC}_{\text{val}}": "ROC-AUC_val",
        r"\text{PR-AUC}_{\text{val}}": "PR-AUC_val",
        r"\text{PR-AUC}_{\text{base}}": "PR-AUC_base",
        r"\text{KS}_{\text{val}}": "KS_val",
        r"\text{MCC}_{\text{Dummy}}": "MCC_Dummy",
        r"\text{Disparate Impact}": "Disparate Impact",
        r"\text{DIR}": "DIR",
        r"\text{FN}": "FN",
        r"\text{FP}": "FP",
        r"\text{TP}": "TP",
        r"\text{TN}": "TN",
        r"\text{TPR}": "TPR",
        r"\text{FPR}": "FPR",
        r"\text{TARGET}": "TARGET",
        r"\text{S/ }": "S/ ",
        r"\text{ s}": " s",
        r"\text{ ms}": " ms",
        r"\text{ pp}": " pp",
        r"\text{ filas}": " filas",
        r"\text{ columnas}": " columnas",
        r"\text{ segundos}": " segundos",
        r"\text{ LGD}": " LGD",
        r"\mid": "|",
        r"\mathbf{x}": "x",
        r"\mathbf{0.7500}": "0.7500",
        r"\mathbf{11.3871}": "11.3871",
        r"\lvert": "|",
        r"\rvert": "|",
        r"\lVert": "||",
        r"\rVert_2^2": "||²",
        r"\mathbf{w}": "w",
        r"\ln\left(\frac{p}{1-p}\right)": "ln(p/(1-p))",
        r"\frac{1}{N}": "(1/N)",
        r"\left(": "(",
        r"\right)": ")",
        r"\in \{1, \dots, 5\}": "∈ {1..5}",
        r"10^{-70}": "10⁻⁷⁰",
        r"e^\beta": "e^β",
        r"\mathbf{x}^\top": "x^T ",
        r"N_{\text{neg}} / N_{\text{pos}}": "N_neg / N_pos",
        r"\Delta\text{AUC}_{\text{train-val}}": "ΔAUC_train-val",
        r"\Delta\text{ROC-AUC}": "ΔROC-AUC",
        r"\sigma_{\text{ROC-AUC}}": "σ_ROC-AUC",
        r"\max_{\tau}": "max_τ",
        r"\text{Costo}(\tau)": "Costo(τ)",
        r"\text{Acc}": "Acc",
        r"\text{ROC-AUC}": "ROC-AUC",
        r"\text{PR-AUC}": "PR-AUC",
        r"\text{KS}": "KS",
        r"\text{MCC}": "MCC",
        r"\text{Recall}": "Recall",
        r"\text{AUC}": "AUC",
        r"\text{--}": "–",
        r"\max_{τ}": "max_τ",
        r"\beta_0": "β_0",
        r"10^{-70}": "10^-70",
    }
    for k, v in sorted(latex_replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(k, v)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = text.replace("&lt;br/&gt;", "<br/>").replace("&lt;br&gt;", "<br/>")

    # Quitar signos $ de expresiones matemáticas inline
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    # Quitar enlaces markdown [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Proteger código inline `code` con placeholders para no alterar * dentro de `...`
    code_spans: list[str] = []

    def _save_code(match: re.Match) -> str:
        idx = len(code_spans)
        code_spans.append(f'<font name="Courier" size="8.2" color="#1e293b">{match.group(1)}</font>')
        return f"@@CODESPAN_{idx}@@"

    text = re.sub(r"`([^`]+)`", _save_code, text)
    # Negrita **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Cursiva *text*
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    # Restaurar código inline
    for idx, html_code in enumerate(code_spans):
        text = text.replace(f"@@CODESPAN_{idx}@@", html_code)
    return text


def build_pdf_report() -> None:
    """Genera `04-modelado/04-MODELADO.pdf` a partir de `04_informe_modelado.md`."""
    print("[build_deliverables] Generando 04-modelado/04-MODELADO.pdf...", flush=True)
    md_path = FASE4_DIR / "04_informe_modelado.md"
    pdf_path = FASE4_DIR / "04-MODELADO.pdf"

    lines = md_path.read_text(encoding="utf-8").splitlines()

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=8,
    )
    style_h2 = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=12,
        spaceAfter=6,
    )
    style_h3 = ParagraphStyle(
        "DocH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=9,
        spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=5,
    )
    style_quote = ParagraphStyle(
        "DocQuote",
        parent=style_body,
        leftIndent=12,
        rightIndent=8,
        backColor=colors.HexColor("#eff6ff"),
        borderPadding=6,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=4,
        spaceAfter=6,
    )
    style_caption = ParagraphStyle(
        "DocCaption",
        parent=style_body,
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor("#4b5563"),
        spaceBefore=3,
        spaceAfter=8,
    )
    style_th = ParagraphStyle(
        "TableHead",
        parent=style_body,
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#111827"),
    )
    style_td = ParagraphStyle(
        "TableCell",
        parent=style_body,
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1f2937"),
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Fase 4: Modelado (CRISP-DM) — Informe Completo — Grupo 12",
        author="Grupo 12 — UNMSM",
    )
    usable_w = A4[0] - 3.2 * cm
    story = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1")))
            story.append(Spacer(1, 6))
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(_format_inline_md(line[2:].strip()), style_title))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Paragraph(_format_inline_md(line[3:].strip()), style_h2))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(_format_inline_md(line[4:].strip()), style_h3))
            i += 1
            continue

        if line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m:
                caption_txt, rel_img = m.group(1), m.group(2)
                img_path = (FASE4_DIR / rel_img).resolve()
                if img_path.exists():
                    with PILImage.open(img_path) as im:
                        iw, ih = im.size
                    max_w = usable_w * 0.92
                    max_h = 8.8 * cm
                    scale = min(max_w / iw, max_h / ih)
                    flow_img = Image(str(img_path), width=iw * scale, height=ih * scale)
                    flow_img.hAlign = "CENTER"
                    story.append(
                        KeepTogether(
                            [
                                Spacer(1, 4),
                                flow_img,
                                Paragraph(_format_inline_md(caption_txt), style_caption),
                            ]
                        )
                    )
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2:
                header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
                data_rows = []
                for tl in table_lines[2:]:
                    row_cells = [c.strip() for c in tl.strip("|").split("|")]
                    if len(row_cells) < len(header_cells):
                        row_cells += [""] * (len(header_cells) - len(row_cells))
                    data_rows.append(row_cells[: len(header_cells)])

                n_cols = len(header_cells)
                # Calcular anchos proporcionales a la longitud máxima de texto por columna
                col_lengths = []
                for col_idx in range(n_cols):
                    texts = [header_cells[col_idx]] + [r[col_idx] for r in data_rows]
                    max_l = max(len(re.sub(r"[`*$_]", "", t)) for t in texts)
                    col_lengths.append(max(max_l, 6))
                # Limitar el peso máximo de una sola columna para no aplastar columnas numéricas
                col_weights = [min(float(l), 38.0) ** 0.72 for l in col_lengths]
                total_w = sum(col_weights)
                col_widths = [usable_w * (w / total_w) for w in col_weights]

                formatted_table = [[Paragraph(_format_inline_md(c), style_th) for c in header_cells]]
                for r in data_rows:
                    formatted_table.append([Paragraph(_format_inline_md(c), style_td) for c in r])

                t = Table(formatted_table, colWidths=col_widths, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                        ]
                    )
                )
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            continue

        if line.startswith(">"):
            quote_text = line.lstrip(">").strip()
            story.append(Paragraph(_format_inline_md(quote_text), style_quote))
            i += 1
            continue

        if re.match(r"^(\*|-|\d+\.)\s+", line):
            items = []
            while i < n and re.match(r"^\s*(\*|-|\d+\.)\s+", lines[i]):
                raw_item = re.sub(r"^\s*(\*|-|\d+\.)\s+", "", lines[i]).strip()
                i += 1
                # Capturar sub-ítems indentados como continuación
                while i < n and lines[i].startswith("   ") and re.match(r"^\s*(\*|-)\s+", lines[i]):
                    sub_item = re.sub(r"^\s*(\*|-)\s+", "", lines[i]).strip()
                    raw_item += f"<br/>• {sub_item}"
                    i += 1
                items.append(
                    ListItem(
                        Paragraph(_format_inline_md(raw_item), style_body),
                        leftIndent=14,
                        bulletColor=colors.HexColor("#1d4ed8"),
                    )
                )
            story.append(ListFlowable(items, bulletType="bullet", start="bulletchar", leftIndent=8))
            story.append(Spacer(1, 3))
            continue

        # Párrafo estándar
        para_lines = [line]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "|", "![", "---", ">"))
            and not re.match(r"^(\*|-|\d+\.)\s+", lines[i].strip())
        ):
            para_lines.append(lines[i].strip())
            i += 1
        story.append(Paragraph(_format_inline_md(" ".join(para_lines)), style_body))

    try:
        doc.build(story)
        print(f"[build_deliverables] PDF generado exitosamente en {pdf_path}", flush=True)
    except PermissionError:
        fallback_pdf = FASE4_DIR / "04-MODELADO_actualizado.pdf"
        print(f"[build_deliverables] {pdf_path.name} está bloqueado por un lector PDF abierto. Guardando en {fallback_pdf.name}...", flush=True)
        doc_fb = SimpleDocTemplate(
            str(fallback_pdf),
            pagesize=A4,
            leftMargin=1.6 * cm,
            rightMargin=1.6 * cm,
            topMargin=1.6 * cm,
            bottomMargin=1.6 * cm,
            title="Fase 4: Modelado (CRISP-DM) — Informe Completo — Grupo 12",
            author="Grupo 12 — UNMSM",
        )
        doc_fb.build(story)
        print(f"[build_deliverables] PDF generado exitosamente en {fallback_pdf}", flush=True)


def cleanup_placeholder() -> None:
    placeholder = FASE4_DIR / "texto"
    if placeholder.exists() and placeholder.stat().st_size == 0:
        placeholder.unlink()
        print("[build_deliverables] Eliminado archivo placeholder vacío 04-modelado/texto", flush=True)


if __name__ == "__main__":
    test_inference_service()
    build_and_execute_notebook()
    build_pdf_report()
    cleanup_placeholder()
    print("[build_deliverables] ¡Todos los entregables de la Fase 4 están completos!", flush=True)
