# 2.2 Descripción de los Datos (con SME) — Home Credit

> Rúbrica 2.2: características para describir datos + consideraciones legales/regulatorias/éticas + informe de descripción. SME consultado: diccionario oficial (SME documentado) + CRO/Riesgos como experto de negocio (requisito: mora = atraso >X días en 1 de las primeras Y cuotas).

## 1. Características por tabla (cantidad + tipos + codificación)

`application_train.csv` (307.511 × 122): 65 `float64`, 41 `int64`, 16 `str` (a convertir a `category` en Fase 3).
- Identidad: `SK_ID_CURR` único, 0 duplicados. Objetivo `TARGET`: 0=282.686 (91.93%), 1=24.825 (8.07%).
- Financieras: `AMT_INCOME_TOTAL` (mediana 147.150, max 117.000.000 outlier), `AMT_CREDIT` (mediana 513.531, max 4.050.000), `AMT_ANNUITY` (12 nulos, mediana 24.903), `AMT_GOODS_PRICE` (278 nulos, mediana 450.000).
- Scores externos: `EXT_SOURCE_1` (56.38% nulos), `_2` (0.21%), `_3` (19.83%). Rangos [0,1].
- Temporales relativos (días antes de solicitud, negativos): `DAYS_BIRTH` (media −16.037 ≈ 43.9 años), `DAYS_EMPLOYED` (centinela 365243 = 18.01%, ver calidad), `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`.
- Categóricas (16): `NAME_CONTRACT_TYPE` (Cash 278.232 / Revolving 29.279), `CODE_GENDER` (F 202.448 / M 105.059 / XNA 4), `FLAG_OWN_CAR/REALTY` (Y/N), `NAME_INCOME_TYPE` (8, incluye `Unemployed` n=22, `Maternity` n=5), `NAME_EDUCATION_TYPE` (5), `NAME_FAMILY_STATUS` (6), `NAME_HOUSING_TYPE` (6), `OCCUPATION_TYPE` (18 + 31.35% nulos), `ORGANIZATION_TYPE` (58), `WEEKDAY_APPR_PROCESS_START` (7), flags `FLAG_*` binarios 0/1.
- Vivienda normalizada (AVG/MODE/MEDI de ~13 conceptos): 48–70% nulos. `OWN_CAR_AGE` 65.99% nulos.
- Codificación: `M/F`, `Y/N`, `Cash loans/Revolving loans`; `FLAG_*` 0/1; `STATUS` en balances usa `C/0/1/2/3/4/5/X` (ver abajo). Sin incoherencias de codificación intra-columna.

`bureau.csv` (1.716.428 × 17): llaves sin nulos. `CREDIT_ACTIVE`: Closed 1.079.273 / Active 630.607 / Sold 6.527 / Bad debt 21. `CREDIT_TYPE`: Consumer 1.251.615 / Credit card 402.195 / resto auto/hipoteca/micro. `CREDIT_DAY_OVERDUE` media 0.82, max 2.792 días. Montos: `AMT_CREDIT_SUM` media 354.995, `AMT_CREDIT_SUM_DEBT` 15.01% nulos, `AMT_CREDIT_SUM_LIMIT` 34.48% nulos, `AMT_ANNUITY` 71.47% nulos.

`bureau_balance.csv` (27.299.925 × 3): sin nulos. `STATUS`: C 13.646.993 / 0 7.499.507 / X 5.810.482 / 1 242.347 / 5 62.406 / 2 23.419 / 3 8.924 / 4 5.847. `MONTHS_BALANCE` [−96, 0], media −30.7 meses de historia.

`previous_application.csv` (1.670.214 × 37): `NAME_CONTRACT_STATUS`: Approved 1.036.781 / Canceled 316.319 / Refused 290.678 / Unused 26.436. `NAME_CONTRACT_TYPE`: Cash 747.553 / Consumer 729.151 / Revolving 193.164. Tasas `RATE_INTEREST_*` 99.64% nulos (inservibles sin tratamiento). `AMT_DOWN_PAYMENT`/`RATE_DOWN_PAYMENT` 53.64% nulos. Fechas `DAYS_*` 40.3% nulos (solicitudes sin desembolso).

`POS_CASH_balance.csv` (10.001.358 × 8): `NAME_CONTRACT_STATUS`: Active 9.151.119 / Completed 744.883 / resto Signed/Demand/Returned. `SK_DPD` media 11.6, max 4.231; `SK_DPD_DEF` media 0.65. `CNT_INSTALMENT(_FUTURE)` 0.26% nulos.

`installments_payments.csv` (13.605.401 × 8): `DPD_PAGO = DAYS_ENTRY − DAYS_INSTALMENT` media −8.8 días (pagan anticipado), 8.43% cuotas con atraso >0; 9.37% pagos con monto menor al pactado. `DAYS_ENTRY_PAYMENT`/`AMT_PAYMENT` 0.02% nulos (cuotas sin pago registrado).

`credit_card_balance.csv` (3.840.312 × 23): `NAME_CONTRACT_STATUS`: Active 3.698.436 / Completed 128.918. `SK_DPD` media 9.28. Utilización `BALANCE/LIMIT` mediana 1.1%, media 37.5% (cola de sobreuso, max 11.8). `AMT_PAYMENT_CURRENT` 20.0% nulos + 6 vars de disposiciones 19.52% nulos (patrón monótono: tarjeta sin movimiento ese mes).

## 2. Consideraciones legales, regulatorias y éticas (checklist SME)

- Privacidad: solo seudónimos `SK_ID_CURR/PREV/BUREAU`. Sin nombres, DNI, teléfono, dirección exacta. Cumple Ley 29733 (Perú) y estándar GDPR/Kaggle.
- Seguridad: dato estático local, sin conexión a producción. No re-identificar ni cruzar con fuentes externas.
- Precisión: `TARGET` definido por Home Credit (regla X/Y días). No auditable contra fuente primaria; se asume exacto a nivel registro.
- Sesgo (hallazgo crítico): `CODE_GENDER` M 10.14% mora vs F 7.00% (paridad F/M 0.69 < 0.8); educación Lower 10.93% vs Academic 1.83% (p=2.4e-219); vivienda alquilada 12.31% vs casa 7.80%. Variables sensibles (`CODE_GENDER`, `DAYS_BIRTH`, educación, región, `ORGANIZATION_TYPE`) solo descriptivas aquí; su uso en modelado exige auditoría de paridad ≥0.8 en Fase 5 (otro grupo).
- Gobernanza: retención académica hasta 08/12/2026; sin despliegue productivo por este grupo.

## 3. Informe de descripción (resumen ejecutivo)

- Formato: 7 CSV planos + diccionario de 219 filas. Método de captura: volcado operacional (no scraping).
- Dimensiones: 307.511 solicitudes × 122 (+ ~55.4M filas relacionales). Cobertura: 99.4% clientes tienen buró, 33.7% tienen tarjeta (subpoblación, no falta aleatoria).
- Estadísticos base calculados para todos los atributos clave (ver §1 y 02_exploracion).
- Atributos priorizados: `EXT_SOURCE_2/3`, `DAYS_BIRTH`, `AMT_ANNUITY/CREDIT`, DPD agregados, `CREDIT_ACTIVE/TYPE`, `NAME_CONTRACT_STATUS` previo. Atributos en observación: vivienda (ausencia informativa), `RATE_INTEREST_*` (descartar), flags casi-constantes.
