# 04 — Criterios de Éxito (PDF Sec 5/10/11)

## Negocio (Sec 5)
- Poder predictivo con desbalance 91.93% / 8.07% → **accuracy inválido**. Exigir **ROC-AUC ≥ 0.75** en validación.
- Automatización ≥ 70% (Business Case afina 75%) con recomendación aprueba/rechaza; resto = zona gris manual.
- Cartera aprobada no excede pérdida máxima de Riesgos (CRO decide).
- Subjetivo: Top 3–5 factores por denegación, lógica económica de patrones, aprobación CRO.

## Minería (Sec 10/11)
- Problema: **clasificación supervisada desbalanceada** → `P(TARGET=1)`.
- Métricas obligatorias: **ROC-AUC, PR-AUC, KS, Matriz confusión, MCC**. Verificado Celda 10: baseline dummy (predecir siempre 0) da accuracy 0.9193 pero ROC=0.5000 → demuestra por qué se descarta accuracy.
- Umbral: **salida si ROC-val < 0.70 tras Fase 5** (Charter). Umbral decisión por **costo esperado**: FN=S/1.800 (3.000×60%), FP=S/150.
- Explicabilidad: SHAP Top-5 por perfil. Equidad: paridad aprobación entre grupos **≥ 0.8** (`lista_variables_sensibles.csv`). Hallazgos Sec 8: género paridad F/M ≈ 0.69 < 0.8 y gradiente educativo 10.93% → 1.83% (p 2.4e-219) → auditorías prioritarias Fase 5.
- Temporal: Fase 5 cierra 17/11/2026; despliegue Fase 6 = prototipo Python + REST + Docker (sin Express), `openapi.yaml`, prueba funcional con respuesta en segundos.

## Alineación criterio ↔ objetivo
Reducir mora → ROC-AUC ≥ 0.75 | Optimizar manual → 70–75% auto | Interpretabilidad regulatoria → SHAP + aprobación CRO.
