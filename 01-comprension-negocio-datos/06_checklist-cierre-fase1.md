# 06 — Checklist Cierre Fase 1 (PDF Sec 12, vence 22/09/2026)

## Negocio — ¿sí?
- [ ] Reducción ≥ 7% pérdidas + 75% auto / zona gris definidos (`00`, `04`)
- [ ] Fin 08/12/2026 con 7 fases + prototipo inferencia (`05`)
- [ ] Presupuesto: S/0 académico / S/189.200 simulado + ROI 126% (Business Case)
- [ ] Acceso 7 tablas + diccionario seudonimizado Ley 29733 (`README`, `02`)
- [ ] 6 riesgos Charter con mitigación (`05`)
- [ ] Costo/beneficio viable (`Business Case`)

## Minería — ¿sí?
- [ ] Clasificación desbalanceada, LightGBM/XGBoost + pesos clase (`04`)
- [ ] Métricas ROC/PR-AUC/KS/MCC + SHAP, umbral ≥ 0.75 (`04`)
- [ ] Despliegue Fase 6 Python+REST+Docker contemplado (`05`)
- [ ] Plan cubre Fase 1→6, riesgos computacionales y no-imputación (`05`, `03`)

## Entregables cronograma Fase 1
- [x] `Business_Case_v3` + `Compresion_de_Negocio.docx` (en DOCUMENTOS, trazados en `00`)
- [x] `diccionario_datos_anotado.csv` (219 filas) — exportar a xlsx para entrega
- [x] `lista_variables_sensibles.csv` — exportar a xlsx para entrega (género 0.69 < 0.8 y gradiente educativo hallados)
- [x] Ampliado `01_EDA_application_train.ipynb` Sec 6-10 (23 celdas: 8.1/8.2/8.3/8.4 + helper centralizado en 8.1) — pendiente solo exportar `reporte_eda_inicial.pdf`
- [x] Repo + entorno (esta carpeta + `.venv` + `requirements`)
- [ ] `reporte_eda_inicial.pdf`: compilar `README + 00–06` + figuras notebook

## Sec 8 completa (8.1/8.2/8.3/8.4: 7 variables, sin pendientes).

Siguiente: Fase 2 `load_data.py` (bloques + downcast) — no iniciar sin exportar `reporte_eda_inicial.pdf`.
