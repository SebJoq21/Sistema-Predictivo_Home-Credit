# ==============================================================================
# Dockerfile — Servicio de Inferencia Home Credit (Fase 4 / Fase 6 CRISP-DM)
# Entorno aislado y reproducible para auditoría técnica (Normativa SBS / Ley 29733)
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    MODEL_FILENAME=modelo_candidato_lgbm.pkl \
    METADATA_FILENAME=metadata_inferencia.json

# Instalar libgomp1 requerido por LightGBM / XGBoost en contenedores Linux slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/inference-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY services/inference-service/main.py /app/main.py

# Usuario no privilegiado por seguridad operativa
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
