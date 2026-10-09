# Plan Fase 02 Comprensión de Datos + Fase 03 Preparación de Datos

**Decisiones del equipo:** (1) notebooks exploratorios + `scripts/*.py` reproducibles, (2) muestra 10% estratificada primero, corrida 100% después, (3) split estratificado después de consolidar.

Referencias: `CLASES/Comprensión de los datos.pdf`, `CLASES/Preparación de los datos.pdf`, `DOCUMENTOS/Cronograma_CRISP-DM_Grupo_12.xlsx` (Fase 2: 23/09–06/10/2026), decisiones vinculantes de `01-comprension-negocio-datos/` (no imputar vivienda/EXT, flags `FALTA_X`, split anti-leakage, paridad ≥ 0.8).

## 1. Estrategia de muestra (contingencia del Charter)

Muestra 10% estratificada por `TARGET` (≈30.751 filas: ~28.269 pago / ~2.483 mora, `random_state=42`, IDs fijos en `data/sample/sample_ids.csv`). Toda la Fase 02–03 se valida primero en muestra; la corrida 100% reutiliza los mismos scripts con flag `--full`.

## 2. Fase 02 — Comprensión de Datos (solo lectura, sin transformar)

| # | Artefacto | Contenido |
|---|---|---|
| 2.1 | `02-comprension-datos/00_recopilacion.md` | 8 CSV, 2.53 GB en disco / >6.6 GB en RAM, llaves (`SK_ID_CURR`, `SK_ID_BUREAU`, `SK_ID_PREV`), PII solo `SK_ID_*` (Ley 29733), `application_test.csv` fuera de alcance |
| 2.2 | `02-comprension-datos/notebooks/01_EDA_bureau.ipynb` | `bureau` (1.716.428 x 17) + `bureau_balance` (27.299.925 x 3, 2 niveles): tipos de crédito, DPD, defaults, historial mensual |
| 2.3 | `02-comprension-datos/notebooks/02_EDA_previous_pos.ipynb` | `previous_application` (1.670.214 x 37) + `POS_CASH_balance` (10.001.358 x 8): estados aprobado/rechazado/cancelado y atrasos vs `TARGET` (descriptivo) |
| 2.4 | `02-comprension-datos/notebooks/03_EDA_pagos_tarjetas.ipynb` | `installments_payments` (13.605.401 x 8) + `credit_card_balance` (3.840.312 x 23): tendencias de pago, uso de tarjeta, mora |
| 2.5 | `02-comprension-datos/01_descripcion_datos.md` | Grano, llave y tipos por tabla; inconsistencia diccionario-vs-real en `previous_application` (38 vs 37) |
| 2.6 | `02-comprension-datos/02_informe-calidad-secundarias.md` | 8 dimensiones: completitud, validez, consistencia, unicidad, integridad referencial, razonabilidad, oportunidad, exactitud |
| 2.7 | `02-comprension-datos/03_hipotesis-features.md` | Agregados candidatos a Fase 03 (promedio DPD, ratio aprobado, tendencias de balance/pago, uso de `EXT_SOURCE`) |

## 3. Fase 03 — Preparación de Datos (scripts reproducibles + notebooks de validación)

Orden estricto; cada script guarda checkpoint en `data/processed/*.parquet` (gitignored, local) y libera RAM (`del` + `gc.collect()`).

| # | Artefacto | Salida |
|---|---|---|
| 3.1 | `03-preparacion-datos/scripts/make_sample.py` | `data/sample/*` commiteable (10% estratificado) |
| 3.2 | `03-preparacion-datos/scripts/load_data.py --sample/--full` | Lectura por bloques + `downcast` (`float64→float32`, `int64→int32/category`), apoyado en `utils/memory_optimization.py` |
| 3.3 | `03-preparacion-datos/scripts/clean_application.py` | `application_train_clean.parquet`: centinela `DAYS_EMPLOYED=365243` → `FLAG_NO_EMPLEADO` (18.01% hallado), topeo documentado de `AMT_INCOME_TOTAL` (max 117M), `ANNUITY`/`GOODS_PRICE` con 12/278 nulos como flag, `str→category`. Sin imputar vivienda ni `EXT_SOURCE` |
| 3.4 | `03-preparacion-datos/scripts/treat_missing.py` | Flags binarios `FALTA_X` + `reporte_faltantes.md` con mecanismo MAR/MNAR por variable (cronograma v3: tratamiento como información, no imputación) |
| 3.5 | `03-preparacion-datos/scripts/agg_bureau.py`, `agg_previous_pos.py`, `agg_cards.py` | `bureau_agg.parquet`, `previous_pos_agg.parquet`, `cc_installments_agg.parquet` — agregación secuencial por `SK_ID_CURR` (mean/max/min/sum/count/tendencias), un checkpoint por tabla |
| 3.6 | `03-preparacion-datos/scripts/build_dataset.py` | `dataset_consolidado.parquet` (1 fila por cliente) + features derivadas (ratios crédito/ingreso, `EDAD_ANIOS`, tendencias de pago, uso de `EXT_SOURCE`) + `lista_features_final.csv` (importancia, correlación, casi-constantes) |
| 3.7 | `03-preparacion-datos/scripts/split_dataset.py` + `notebooks/validacion_split.ipynb` | `train.parquet`/`val.parquet` estratificados **después** de consolidar (agregados solo con historia pasada) + `checklist_leakage.md` + `anexo_auditoria_preliminar.md` (paridad género 0.69 y gradiente educativo como focos) |
| 3.8 | `requirements-dev.txt` | Añadir `pyarrow` (única dependencia nueva, para Parquet) |

## 4. Reglas vinculantes (heredadas de Fase 1)

1. No joins a nivel transacción para modelar; todo a nivel `SK_ID_CURR`.
2. No imputar vivienda (~60–70% nulos) ni `EXT_SOURCE_1` (56.4%) con media/moda; flags + manejo nativo NaN (LightGBM/XGBoost).
3. Split estratificado train/val antes de cualquier estadística que cruce el corte (anti-leakage).
4. Sensibles (`CODE_GENDER`, `DAYS_BIRTH`, educación, región) solo con auditoría de paridad ≥ 0.8.
5. `data/processed/*.parquet` no se commitea; `data/sample/` sí.

## 5. Commits propuestos

1. `docs(fase2): 02 comprensión de datos + hipótesis de features`
2. `feat(fase3): sample + carga + limpieza + tratamiento de faltantes`
3. `feat(fase3): agregados + dataset consolidado + split + leakage`
