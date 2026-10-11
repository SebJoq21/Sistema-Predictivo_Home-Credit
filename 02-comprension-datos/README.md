# 02 — Comprensión de Datos (CRISP-DM Fase 2) — Grupo 12 (solo esta fase)

> Rúbrica parcial 2.1–2.4. Teoría: `Sesiones/4/`. Datos: `DATASET/` (7 CSV + diccionario, `ARCHIVOS QUE NO UTILIZAREMOS/` excluido).

| Informe | Rúbrica | Contenido |
|---|---|---|
| `00_recopilacion.md` | 2.1 | Orígenes, fuentes, inventario verificado (filas/llaves/grano) |
| `01_descripcion_datos.md` | 2.2 | Cantidad, tipos, codificación + checklist legal/ética Ley 29733 + sesgos |
| `02_exploracion_EDA.md` | 2.3 | Continuos, categóricos (chi2), ordinales (Spearman), uni/multi/dimensional + 6 hipótesis |
| `03_calidad_datos.md` | 2.4 | 8 dimensiones + plan de saneamiento para el grupo de Preparación |

## Cifras ancla (verificadas por código 11/10/2026)
- Principal 307.511 × 122, mora 8.07% (24.825), `SK_ID_CURR` único, 0 duplicados.
- Secundarias: 1.7M buró + 27.3M balance + 1.67M previas + 10M POS + 13.6M cuotas + 3.84M tarjetas ≈ 2.66 GB.
- Top señal: `EXT_SOURCE_3` r=−0.179, `_2` −0.160, `_1` −0.155. Sesgo género paridad 0.69 <0.8.
- Nulos críticos: vivienda 48–70%, `EXT_SOURCE_1` 56.38%, `AMT_ANNUITY` buró 71.47%, `RATE_INTEREST` 99.64% (descartar).
- Inconsistencia: diccionario 38 vs real 37 en `previous_application`.

## Cierre Fase 2 (checklist)
- [x] Orígenes/fuentes/informe recopilación
- [x] Descripción + SME (diccionario + CRO) + legal/ética
- [x] EDA completo con chi2/Pearson/Spearman + dimensional
- [x] Calidad 8 dimensiones + plan para Fase 3 (otro grupo)
- [ ] Exportar a PDF con figuras (barras, boxplots, heatmap correlación) + tablas IEEE para entrega
