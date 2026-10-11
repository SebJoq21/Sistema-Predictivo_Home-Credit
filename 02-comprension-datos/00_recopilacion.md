# 2.1 Recopilación de Datos Iniciales — Home Credit Default Risk

> Alcance del grupo: solo Comprensión de Datos (CRISP-DM Fase 2). No se transforma ni imputa. Fuente teórica: `Sesiones/4/Comprensión de los datos.pdf` + `Sesiones/4/Minería de Datos - Sesión 4.pdf`. Rúbrica 2.1: orígenes, fuentes, informe de recopilación.

## 1. Orígenes (clasificación Sesión 4)

| Origen | Aplica | Detalle |
|---|---|---|
| Datos existentes (transaccionales) | SÍ, 100% | Las 7 tablas son histórico operacional de Home Credit + burós externos. Suficientes para el objetivo: predecir `TARGET`. |
| Datos adquiridos (demográficos externos) | PARCIAL | Columnas `EXT_SOURCE_1/2/3` ya vienen importadas en `application_train.csv`. No se compra nada adicional. |
| Datos adicionales (encuestas/seguimiento) | NO | No se requiere. Prohibido por alcance: solo lo entregado en `DATASET/`. |

Preguntas guía Sesión 4 (respuesta):
- ¿Atributos prometedores? `EXT_SOURCE_2/3`, `DAYS_BIRTH`, `AMT_CREDIT/ANNUITY`, historial DPD en `installments/POS/bureau_balance`, `CREDIT_ACTIVE/TYPE`.
- ¿Atributos a excluir (candidatos)? `RATE_INTEREST_PRIMARY/PRIVILEGED` (99.64% nulos), `FLAG_MOBIL/CONT_MOBILE` (casi constantes), `FLAG_EMAIL` no significativo (p=0.34).
- ¿Datos suficientes? Sí: 307.511 solicitudes, 24.825 morosos (8.07%) → potencia suficiente para clasificación desbalanceada.
- ¿Fusión con problemas? Sí: 2 niveles (`bureau_balance → bureau → application`) y 3 tablas vía `SK_ID_PREV`. Todo debe agregarse a nivel `SK_ID_CURR` en Fase 3, nunca join a nivel transacción.
- ¿Nulos por origen? Ver informe 03_calidad: vivienda 48–70%, `EXT_SOURCE_1` 56.38%, `AMT_ANNUITY` en buró 71.47%.

## 2. Fuentes (clasificación Sesión 4)

| Fuente | Archivos | Formato | Acceso |
|---|---|---|---|
| Flat Files (CSV) | 7 + diccionario | CSV `,` + header único, sin fila total, fechas como enteros relativos (`DAYS_*`, `MONTHS_BALANCE`) | Local `DATASET/`, lectura con `pandas.read_csv`, preprocesamiento medio |
| Database / Warehouse / Lake / API / Scraping | — | — | NO aplica. No hay BD viva ni API. Dato estático, documentado en `HomeCredit_columns_description.csv` (219 filas). |

Verificación formato: cada CSV tiene 1 fila de header, mismo nº de campos por registro, delimitador `,` consistente. Falla detectada: diccionario declara 38 vars para `previous_application` pero el CSV real trae 37 (ver 03_calidad).

## 3. Informe de recopilación (inventario verificado por código)

Medición directa (conteo exacto por chunks, 11/10/2026):

| # | Archivo | Filas | Cols | Llaves | Peso disco | Grano |
|---|---|---|---|---|---|---|
| 1 | `application_train.csv` | 307.511 | 122 | `SK_ID_CURR` (única) | 166 MB | 1 fila = 1 solicitud |
| 2 | `bureau.csv` | 1.716.428 | 17 | `SK_ID_CURR, SK_ID_BUREAU` | 170 MB | 1 fila = 1 crédito en otro buró (~5.61 por cliente) |
| 3 | `bureau_balance.csv` | 27.299.925 | 3 | `SK_ID_BUREAU` | 376 MB | 1 fila = 1 mes de crédito buró |
| 4 | `previous_application.csv` | 1.670.214 | 37 | `SK_ID_CURR, SK_ID_PREV` | 405 MB | 1 fila = 1 solicitud previa (~4.93 por cliente con historial) |
| 5 | `POS_CASH_balance.csv` | 10.001.358 | 8 | `SK_ID_CURR, SK_ID_PREV` | 393 MB | 1 fila = 1 mes POS/cash (~29.66 por cliente) |
| 6 | `installments_payments.csv` | 13.605.401 | 8 | `SK_ID_CURR, SK_ID_PREV` | 723 MB | 1 fila = 1 pago/cuota (~40.06 por cliente) |
| 7 | `credit_card_balance.csv` | 3.840.312 | 23 | `SK_ID_CURR, SK_ID_PREV` | 425 MB | 1 fila = 1 mes tarjeta (~37.08 por cliente) |
| 8 | `HomeCredit_columns_description.csv` | 219 | 5 | — | 37 KB | 122 app + 38 prev (diccionario) + 23 cc + 17 bureau + 8 POS + 8 inst + 3 balance |

Total ≈ 2.66 GB en disco. `ARCHIVOS QUE NO UTILIZAREMOS/` (`application_test.csv`, `sample_submission.csv`) queda fuera de alcance por Charter.

Decisión para Fase 3 (solo se deja planteada, no se ejecuta): lectura por bloques + `downcast float64→float32, int64→int32/category` + agregación por `SK_ID_CURR` a Parquet, tabla por tabla.
