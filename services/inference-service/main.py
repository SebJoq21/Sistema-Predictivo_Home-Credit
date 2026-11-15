"""
Servicio de Inferencia de Riesgo Crediticio — Home Credit (Grupo 12).
Fase 4 CRISP-DM (`Cronograma` Fila 25): Desarrollo inicial (borrador funcional) con FastAPI.
Preparado para consolidación en Fase 6 (Implementación Analítica).

Cumplimiento normativo:
- Ley N.° 29733 (Protección de Datos Personales - Perú): Opera exclusivamente sobre `sk_id_curr`
  (seudonimizado), sin recibir ni registrar nombres, DNI, dirección o teléfono en logs.
- Normativa SBS (Equidad y Transparencia): Excluye las variables protegidas (`CODE_GENDER`,
  `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`) del vector de inferencia, aplica segmentación
  tricotómica (APROBAR / ZONA_GRIS_REVISION_MANUAL / RECHAZAR) y devuelve los factores principales.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuración de logs sin PII (Ley N.° 29733)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | inference-service | %(message)s",
)
logger = logging.getLogger("inference-service")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR)))
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "modelo_candidato_lgbm.pkl")
METADATA_FILENAME = os.getenv("METADATA_FILENAME", "metadata_inferencia.json")

app = FastAPI(
    title="Home Credit Risk Scoring — Inference API (Grupo 12)",
    description=(
        "Prototipo de servicio de inferencia en Python (FastAPI) para estimar la probabilidad "
        "de incumplimiento P(TARGET=1), segmentar la zona gris de revisión manual y exponer "
        "los factores determinantes del perfil crediticio."
    ),
    version="0.4.0-fase4",
)

_STATE: Dict[str, Any] = {
    "model": None,
    "metadata": None,
    "loaded_at": None,
}


def load_artifacts() -> None:
    """Carga en memoria el modelo serializado y los metadatos de las 240 variables."""
    model_path = MODELS_DIR / MODEL_FILENAME
    meta_path = MODELS_DIR / METADATA_FILENAME

    if not model_path.exists() or not meta_path.exists():
        logger.warning(
            "Artefactos aún no encontrados en %s (%s, %s).",
            MODELS_DIR,
            MODEL_FILENAME,
            METADATA_FILENAME,
        )
        return

    _STATE["model"] = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        _STATE["metadata"] = json.load(f)
    _STATE["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info(
        "Modelo cargado exitosamente: %s (%d variables autorizadas).",
        MODEL_FILENAME,
        _STATE["metadata"].get("n_features", 0),
    )


@app.on_event("startup")
def on_startup() -> None:
    load_artifacts()


classCreditApplicationRequest = BaseModel


class CreditScoringRequest(BaseModel):
    sk_id_curr: int = Field(
        ...,
        gt=0,
        description="Identificador único seudonimizado de la solicitud (Ley N.° 29733).",
        examples=[100002],
    )
    features: Dict[str, Optional[Any]] = Field(
        ...,
        description="Diccionario clave-valor con las variables del perfil consolidado del solicitante.",
    )


class BatchCreditScoringRequest(BaseModel):
    applications: List[CreditScoringRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Lote de solicitudes a calificar (máximo 500 por petición).",
    )


class CreditScoringResponse(BaseModel):
    sk_id_curr: int
    probabilidad_impago: float
    decision: str
    requiere_revision_manual: bool
    umbral_optimo_costo: float
    rango_zona_gris: List[float]
    modelo_utilizado: str
    top_factores_modelo: List[Dict[str, Any]]
    latencia_ms: float


def _prepare_dataframe(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Alinea el payload de entrada exactamente a las 240 variables autorizadas,
    descartando cualquier atributo protegido (`CODE_GENDER`, `NAME_EDUCATION_TYPE`,
    `NAME_FAMILY_STATUS`) o campo extraño (prevención de inyección de atributos).
    """
    feature_names: List[str] = meta["feature_names"]
    cat_cols: List[str] = meta["categorical_features"]
    cat_domains: Dict[str, List[str]] = meta["categorical_domains"]
    num_cols: List[str] = meta["numerical_features"]

    rows = []
    for rec in records:
        row = {col: rec.get(col, np.nan) for col in feature_names}
        rows.append(row)

    df = pd.DataFrame(rows, columns=feature_names)

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    for c in cat_cols:
        vals = df[c].where(df[c].notna(), None)
        vals = [str(v) if v is not None else np.nan for v in vals]
        df[c] = pd.Categorical(vals, categories=cat_domains.get(c, []))

    return df


def _classify_decision(prob: float, thr_info: Dict[str, float]) -> tuple[str, bool]:
    low_grey = float(thr_info.get("grey_zone_low_preliminary", 0.35))
    high_grey = float(thr_info.get("grey_zone_high_preliminary", 0.60))
    if prob < low_grey:
        return "APROBAR_AUTOMATICO", False
    if prob <= high_grey:
        return "ZONA_GRIS_REVISION_MANUAL", True
    return "RECHAZAR_AUTOMATICO", False


@app.get("/health", tags=["Operaciones"])
def health_check() -> Dict[str, Any]:
    if _STATE["model"] is None:
        load_artifacts()
    ready = _STATE["model"] is not None and _STATE["metadata"] is not None
    return {
        "status": "ok" if ready else "degraded_model_not_loaded",
        "service": "home-credit-inference-service",
        "phase": "Fase 4 - Modelado (Borrador Funcional)",
        "model_loaded": ready,
        "model_name": MODEL_FILENAME if ready else None,
        "loaded_at": _STATE["loaded_at"],
    }


@app.get("/model/info", tags=["Gobernanza y Metadatos"])
def model_info() -> Dict[str, Any]:
    if _STATE["metadata"] is None:
        load_artifacts()
    if _STATE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Metadatos del modelo no disponibles.")
    meta = _STATE["metadata"]
    return {
        "model_name": meta["model_name"],
        "algorithm": meta["algorithm"],
        "phase": meta["phase"],
        "n_features": meta["n_features"],
        "n_numerical": len(meta["numerical_features"]),
        "n_categorical": len(meta["categorical_features"]),
        "protected_excluded_features": meta["protected_excluded_features"],
        "thresholds": meta["thresholds"],
        "validation_metrics": meta["validation_metrics"],
        "top_10_features_gain": meta["top_10_features_gain"],
    }


@app.post("/predict", response_model=CreditScoringResponse, tags=["Inferencia"])
def predict_single(req: CreditScoringRequest) -> CreditScoringResponse:
    t0 = time.perf_counter()
    if _STATE["model"] is None or _STATE["metadata"] is None:
        load_artifacts()
    if _STATE["model"] is None or _STATE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Modelo candidato no cargado en el servidor.")

    meta = _STATE["metadata"]
    model = _STATE["model"]

    df_input = _prepare_dataframe([req.features], meta)
    prob = float(model.predict_proba(df_input)[:, 1][0])

    thr_info = meta["thresholds"]
    decision, is_grey = _classify_decision(prob, thr_info)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    # Registro seguro sin PII (solo SK_ID_CURR y decisión)
    logger.info(
        "score_request | sk_id_curr=%d | prob=%.4f | decision=%s | latency_ms=%.2f",
        req.sk_id_curr,
        prob,
        decision,
        latency_ms,
    )

    return CreditScoringResponse(
        sk_id_curr=req.sk_id_curr,
        probabilidad_impago=round(prob, 5),
        decision=decision,
        requiere_revision_manual=is_grey,
        umbral_optimo_costo=float(thr_info["optimal_expected_cost"]),
        rango_zona_gris=[
            float(thr_info["grey_zone_low_preliminary"]),
            float(thr_info["grey_zone_high_preliminary"]),
        ],
        modelo_utilizado=meta["model_name"],
        top_factores_modelo=meta["top_10_features_gain"][:5],
        latencia_ms=latency_ms,
    )


@app.post("/predict/batch", tags=["Inferencia"])
def predict_batch(req: BatchCreditScoringRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if _STATE["model"] is None or _STATE["metadata"] is None:
        load_artifacts()
    if _STATE["model"] is None or _STATE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Modelo candidato no cargado en el servidor.")

    meta = _STATE["metadata"]
    model = _STATE["model"]

    df_input = _prepare_dataframe([item.features for item in req.applications], meta)
    probs = model.predict_proba(df_input)[:, 1]
    thr_info = meta["thresholds"]

    results = []
    for item, p in zip(req.applications, probs):
        prob_val = float(p)
        decision, is_grey = _classify_decision(prob_val, thr_info)
        results.append(
            {
                "sk_id_curr": item.sk_id_curr,
                "probabilidad_impago": round(prob_val, 5),
                "decision": decision,
                "requiere_revision_manual": is_grey,
            }
        )

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    logger.info("batch_score_request | count=%d | latency_ms=%.2f", len(results), latency_ms)
    return {
        "total_evaluados": len(results),
        "modelo_utilizado": meta["model_name"],
        "latencia_total_ms": latency_ms,
        "resultados": results,
    }
