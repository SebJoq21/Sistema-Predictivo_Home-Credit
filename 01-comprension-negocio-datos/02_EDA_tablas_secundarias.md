# 02 — EDA Tablas Secundarias (PDF Sec 6.2, verificado en repo)

Conteos verificados hoy por conteo directo de filas (sin cargar a RAM). Columnas por header. Llaves por header.

| Tabla | Filas datos | Cols | Llaves | Peso disco | Grano | Join a cliente |
|---|---|---|---|---|---|---|
| `bureau.csv` | 1.716.428 | 17 | `SK_ID_CURR, SK_ID_BUREAU` | 162.1 MB | 1 fila = 1 crédito en otro buró (~5.6 por cliente) | directo `SK_ID_CURR` |
| `bureau_balance.csv` | 27.299.925 | 3 | `SK_ID_BUREAU` | 358.2 MB | 1 fila = 1 mes de un crédito buró | 2 niveles: `BALANCE.SK_ID_BUREAU → BUREAU.SK_ID_CURR` |
| `previous_application.csv` | 1.670.214 | 37 | `SK_ID_CURR, SK_ID_PREV` | 386.2 MB | 1 fila = 1 solicitud previa (~5.4 por cliente con historial) | directo `SK_ID_CURR` |
| `POS_CASH_balance.csv` | 10.001.358 | 8 | `SK_ID_CURR, SK_ID_PREV` | 374.5 MB | 1 fila = 1 mes POS/cash | vía `SK_ID_PREV` → agregar por cliente |
| `installments_payments.csv` | 13.605.401 | 8 | `SK_ID_CURR, SK_ID_PREV` | 689.6 MB | 1 fila = 1 pago en cuotas | vía `SK_ID_PREV` → agregar por cliente |
| `credit_card_balance.csv` | 3.840.312 | 23 | `SK_ID_CURR, SK_ID_PREV` | 404.9 MB | 1 fila = 1 mes tarjeta | vía `SK_ID_PREV` → agregar por cliente |

Total secundarias en disco ≈ 2.375 GB + principal 158 MB ≈ 2.53 GB (≈2.6 GB del Business Case). En RAM crudo > 6.6 GB (coherente con PDF Sec 6.1: exigir 16 GB mín, 32 ideal).

## Implicancias para Fase 2/3
- Estrategia: leer por bloques + `downcast (float64→float32, int64→int32/category)` + agregar por cliente + guardar `*.parquet` por tabla antes de pasar a la siguiente (`bureau_agg`, `previous_pos_agg`, `cc_installments_agg` según cronograma).
- No hacer joins a nivel transacción para modelar; todo a nivel `SK_ID_CURR` (perfil de comportamiento por cliente).
- `previous_application.csv` declara 38 vars en diccionario pero trae 37 cols reales — registrar en informe calidad como inconsistencia diccionario-vs-real.
- `bureau_balance` (27.3M) e `installments` (13.6M) son el cuello de botella: procesar secuencial por lotes + `gc.collect()`, muestra 10% solo como contingencia de tiempo.
