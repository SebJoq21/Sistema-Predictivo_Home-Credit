# 2.4 Verificación de Calidad de Datos — 8 Dimensiones (Sesión 4)

> Medición exacta por código (conteo chunk 300k) + EDA. Regla Fase 2: solo diagnosticar y planificar; la limpieza la ejecuta otro grupo (Fase 3).

## 1. Exactitud
- `TARGET` 8.07% coherente con mora publicada (~8%). Sin fuente primaria para re-verificar registro a registro; se acepta como verdad terreno.
- Outliers exactos: `AMT_INCOME_TOTAL` max 117.000.000 (684× la media, probable error de digitación o caso extremo); `AMT_ANNUITY` max 258.026; `CREDIT_DAY_OVERDUE` max 2.792 días; `SK_DPD` POS max 4.231 días. Se marcan como ruido/atípico a topear en Fase 3, no se eliminan aquí.
- Centinela `DAYS_EMPLOYED=365243` (55.374, 18.01%) no es error de medición sino código de "sin empleo formal" → exactitud condicional, exige flag.

## 2. Completitud (% nulos exactos)
- `application_train` (67/122 con nulos): vivienda 48.78–69.87% (`COMMONAREA` 69.87%, `NONLIVINGAPARTMENTS` 69.43%, `LIVINGAPARTMENTS` 68.35%, `FLOORSMIN` 67.85%, `YEARS_BUILD` 66.50%, `OWN_CAR_AGE` 65.99%, `LANDAREA` 59.38%); `EXT_SOURCE_1` 56.38%, `_3` 19.83%, `_2` 0.21%; `OCCUPATION_TYPE` 31.35%; `AMT_GOODS_PRICE` 0.09% (278), `AMT_ANNUITY` 12, `NAME_TYPE_SUITE` 0.42%.
- `bureau`: `AMT_ANNUITY` 71.47%, `AMT_CREDIT_MAX_OVERDUE` 65.51%, `DAYS_ENDDATE_FACT` 36.92%, `AMT_CREDIT_SUM_LIMIT` 34.48%, `AMT_CREDIT_SUM_DEBT` 15.01%, `DAYS_CREDIT_ENDDATE` 6.15%.
- `previous_application`: `RATE_INTEREST_*` 99.64% (inservible), `AMT_DOWN_PAYMENT`/`RATE_DOWN_PAYMENT` 53.64%, `NAME_TYPE_SUITE` 49.12%, bloque `DAYS_*`/`NFLAG_INSURED` 40.30%, `AMT_GOODS_PRICE` 23.08%, `AMT_ANNUITY`/`CNT_PAYMENT` 22.29%.
- `credit_card`: `AMT_PAYMENT_CURRENT` 20.0%, 6 vars disposiciones 19.52% (patrón monótono mensual), `AMT_INST_MIN_REGULARITY`/`CNT_INSTALMENT_MATURE_CUM` 7.95%.
- `POS`: `CNT_INSTALMENT(_FUTURE)` 0.26%. `installments`: `DAYS_ENTRY_PAYMENT`/`AMT_PAYMENT` 0.02%. `bureau_balance`: 0% nulos.
- Plan (para otro grupo): nulos vivienda/vehículo/scores = ausencia informativa → flag `FALTA_X` + algoritmo nativo NaN; nunca media/moda. `RATE_INTEREST_*` candidata a exclusión.

## 3. Consistencia
- Diccionario-vs-real: `previous_application` 38 declaradas vs 37 reales. Formato `,` y nº campos consistente en los 7 CSV.
- Codificación consistente intra-columna (`Y/N`, `M/F`, `C/0/X/1-5`). Sin mezcla `M/masculino`.
- Consistencia entre tablas: `SK_ID_CURR` numérico en todas; `STATUS` usa mismo dominio en `bureau_balance`; `NAME_CONTRACT_TYPE/STATUS` dominios compatibles entre `application` y `previous`.

## 4. Integridad (referencial + interna)
- Referencial: `bureau` cubre 305.811/307.511 train (99.45%, 1.700 sin buró = clientes sin historia externa, válido). `credit_card` solo 103.558 (33.68%, subpoblación con tarjeta). `previous/POS/installments` traen ~338–339k únicos porque incluyen `test` (filtrar por train en Fase 3; 0 huérfanos tras filtro).
- Interna: 0 filas totalmente vacías; 1 fila header por archivo; sin registros truncados.
- Claves: `SK_ID_CURR` train única (307.511/307.511); `SK_ID_BUREAU` única en `bureau`; `SK_ID_PREV` única en `previous`. Tablas de balance/pagos no exigen unicidad por diseño (grano mensual/pago).

## 5. Razonabilidad
- Tasas por segmento tienen sentido negocio: alquilados 12.31% > propietarios casa 7.80%; cash 8.35% > revolving 5.48%; rating regional 1→4.8% vs 3→11.1%.
- Conflictos aparentes a auditar: `Unemployed` n=22 con mora 36% pero `FLAG_NO_EMPLEADO` (55k) con mora 5.4% → el centinela no significa desempleado (probable jubilado). `AMT_INCOME_TOTAL` 117M con `TARGET=0/1` a revisar caso a caso. `ORGANIZATION_TYPE=XNA` 55.374 coincide exactamente con centinela → coherente (sin empleador registrado).
- Distribuciones temporales `DAYS_*` negativas y `MONTHS_BALANCE` [−96,0] razonables (relativos a solicitud).

## 6. Oportunidad (vigencia)
- Datos estáticos de competencia 2018, sin timestamp de extracción. Volatilidad esperada alta (comportamiento crediticio cambia) → vigencia vencida para producción, válida para fines académicos.
- Historia por cliente: buró hasta 96 meses, promedio 30.7 → suficiente profundidad temporal. No hay lagunas de muestreo mensual (serie continua por crédito).

## 7. Unicidad / Deduplicación
- `application_train`: 0 duplicados exactos, 0 `SK_ID_CURR` repetidos, 0 nulos en llave/`TARGET`.
- Secundarias: duplicados exactos no esperados por grano; duplicidad de llave solo válida en balances (múltiples meses por crédito). Verificación pendiente para Fase 3: test de unicidad `SK_ID_BUREAU` y `SK_ID_PREV`.

## 8. Validez (dominio)
- Rangos válidos: `EXT_SOURCE_*` [0,1]; `FLAG_*` {0,1}; `CNT_CHILDREN` [0,19] (19 hijos roza límite, 2 casos); `CODE_GENDER` {M,F,XNA} con XNA n=4 (dominio residual válido pero no modelable); `CREDIT_ACTIVE` {Closed,Active,Sold,Bad debt}.
- Tipos: 65 float + 41 int + 16 str en principal, sin strings en columnas numéricas. `NUM_INSTALMENT_VERSION` leída como float por nulos implícitos (validar en Fase 3).
- Precisión: montos con 1 decimal, scores con 3–6 decimales, suficiente.

### Plan de saneamiento (entrega al grupo de Preparación)
1. Flags `FALTA_X` para vivienda/scores/vehículo; 2. `FLAG_NO_EMPLEADO` para 365243; 3. Excluir `RATE_INTEREST_*`; 4. Topeo documentado de `AMT_INCOME_TOTAL` 117M; 5. Filtrar secundarias por `SK_ID_CURR` train; 6. `str→category`; 7. Checklist leakage (split antes de agregar).
