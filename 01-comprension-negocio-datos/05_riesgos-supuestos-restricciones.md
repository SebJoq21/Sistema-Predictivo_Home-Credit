# 05 — Recursos, Requisitos, Riesgos (PDF Sec 6/7/8)

## Inventario verificado (Sec 6)
- **Hardware**: mín 16 GB RAM (ideal 32), CPU multinúcleo, 15 GB SSD. Justificación: secundarias > 6.6 GB en RAM crudo, 2.53 GB en disco verificados.
- **Datos**: `../data/raw/` local, estático, sin OLTP en vivo. 7 tablas + diccionario. Sin compra externa. PII eliminada, solo `SK_ID_*` (Ley 29733, GDPR).
- **Personal**: Chicata (minería), Dvořák (CRO sponsor/validador), Volkova (CDO proveedora datos).
- **Entorno**: `.venv` Python 3.13.5, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1 (chi2 + métricas verificados).

## Requisitos/supuestos/restricciones (Sec 7)
- Legal: no re-identificar, no usar protegidas para decidir, validación entrada anti-inyección, sin PII en logs.
- Plan: 6 fases CRISP-DM + cierre 08/12/2026.
- Despliegue: pipeline batch/API que devuelve `P(impago)` a Underwriting, no frontend ni core bancario.
- Supuestos: costo limitante = cómputo; nulos = ausencia informativa; CRO exige Feature Importance/SHAP.
- Restricciones comprobadas: acceso total local sin VPN; anonimización OK; presupuesto = capacidad local (`downcast + gc.collect()`).

## Riesgos y contingencia (Sec 8)
| Riesgo | Contingencia |
|---|---|
| Cronograma: Feature Engineering 6 tablas excede tiempo | batch secuencial + checkpoints disco; muestra 10% primero si apremia |
| Financiero: RAM satura → nube AWS/GCP | `float32/int32/category`, `del + gc.collect()`, agregación previa, Parquet |
| Datos: 70% nulos degrada aprendizaje | sin imputación artificial; LightGBM/XGBoost nativo + flags `FALTA_X` |
| Resultados: ROC < 0.75 | ensemble / segmentación no supervisada previa; salida si < 0.70 tras Fase 5 |
| + Charter: leakage, sesgo, drift realidad | split previo + CV estratificada; auditoría sesgo (género 0.69 < 0.8 y gradiente educativo 10.93→1.83% hallados Sec 8); features replicables en producción |
| Entorno: `Pandas4Warning` str/object | mitigado con `select_dtypes(include=['str','object','string'])` + `category` (Celdas 6-9) |
