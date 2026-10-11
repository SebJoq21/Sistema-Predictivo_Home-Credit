# 2.3 Exploración de Datos (EDA) — Home Credit

> Rúbrica 2.3: continuos + categóricos + ordinales + univariado + multivariado + dimensional + informe. Cálculos ejecutados en Python (pandas 3.0.5 + scipy 1.18.1) sobre `DATASET/` local. Chi-cuadrado con p<0.05 = dependencia significativa.

## 1. Datos continuos (univariado)

| Variable | Mediana | Media | p99 / Max | Lectura |
|---|---|---|---|---|
| `AMT_INCOME_TOTAL` | 147.150 | 168.798 | 472.500 / 117.000.000 | Cola derecha extrema; 1 outlier 117M (documentar, no remover aquí) |
| `AMT_CREDIT` | 513.531 | 599.026 | 1.850.000 / 4.050.000 | Monto típico ~3.3× ingreso anual |
| `AMT_ANNUITY` | 24.903 | 27.109 | 70.006 / 258.026 | Carga mensual típica ~16–17% del ingreso |
| `AMT_GOODS_PRICE` | 450.000 | 538.396 | — / 4.050.000 | 278 nulos (0.09%) |
| `EXT_SOURCE_1/2/3` | 0.51/0.57/0.54 | 0.50/0.51/0.51 | [0,1] | Distribución acampanada; `_1` solo 134.133 observados |
| `DAYS_BIRTH` | −15.750 (≈43.1 años) | −16.037 | — | Morosos ~3.4 años más jóvenes |
| `DAYS_EMPLOYED` | −1.213 | 63.815 (sesgada por centinela) | — / 365243 | Bimodal por centinela 18.01% |
| DPD `installments` (derivado) | −6 días | −8.8 | 8.43% atrasos >0 | Mayoritariamente puntuales/adelantados |
| Utilización tarjeta | 1.1% | 37.5% | max 11.8× límite | Minoría sobreendeudada |

Boxplots/histogramas (a generar como figuras): confirman que ratios `CREDITO/INGRESO` (pago 3.27 vs mora 3.25) no separan clases por sí solos.

## 2. Datos categóricos (bivariado vs TARGET, chi2)

Todas p << 0.001 salvo `WEEKDAY` (p=0.017), `FLAG_EMAIL` (p=0.34, no significativa), `FLAG_CONT_MOBILE` (p=0.90), `FLAG_MOBIL` (constante, p=1.0).

| Variable | Peor categoría (mora) | Mejor categoría | p chi2 |
|---|---|---|---|
| `NAME_CONTRACT_TYPE` | Cash 8.35% | Revolving 5.48% | 1.0e-65 |
| `CODE_GENDER` | M 10.14% | F 7.00% (XNA n=4) | 1.1e-200 |
| `FLAG_OWN_CAR` | N 8.50% | Y 7.24% | 9.3e-34 |
| `FLAG_OWN_REALTY` | N 8.32% | Y 7.96% | 6.7e-04 (leve pero significativa) |
| `NAME_INCOME_TYPE` | Unemployed 36.4% (n=22), Maternity 40% (n=5) | Pensioner 5.39%, State servant 5.75% | 1.9e-266 |
| `NAME_EDUCATION_TYPE` | Lower 10.93% | Academic 1.83% (gradiente perfecto) | 2.4e-219 |
| `NAME_FAMILY_STATUS` | Civil 9.94%, Single 9.81% | Widow 5.82% | 7.7e-107 |
| `NAME_HOUSING_TYPE` | Rented 12.31%, With parents 11.70% | Office 6.57% | 1.1e-88 |
| `OCCUPATION_TYPE` | Low-skill 17.15%, Drivers 11.33%, Laborers 10.58% | Accountants 4.83%, Managers 6.21% | ~0 |
| `ORGANIZATION_TYPE` | Transport:3 15.75%, Restaurant 11.71% | Trade:4 3.12%, Industry:12 3.79% | 5.2e-299 |
| `FLAG_EMP_PHONE` | con teléfono empresa 8.66% | sin 5.40% (correlato de `FLAG_NO_EMPLEADO`) | 2.5e-143 |

Gráficos: barras apiladas por categoría + heatmap de tasas. Tablas de frecuencia de dos vías en anexo.

## 3. Datos ordinales (Spearman + tendencia)

- `REGION_RATING_CLIENT`: 1→4.82% / 2→7.89% / 3→11.10% mora (monótona).
- `REGION_RATING_CLIENT_W_CITY`: 1→4.84% / 2→7.92% / 3→11.40% (monótona, confirma).
- `HOUR_APPR_PROCESS_START`: nocturnas 0–7h mora 8–15% vs tarde 15–17h ~6.5–7.6% (posible proxy de canal, no causal).
- `CNT_CHILDREN`: 0→7.71% / 1→8.92% / 2→8.72% / 3→9.63% / 4→12.82% (tendencia, n pequeño en 5+).
- Spearman vs TARGET confirma Pearson: `EXT_SOURCE_3` −0.166, `_1` −0.151, `_2` −0.147, `DAYS_BIRTH` +0.078, `DAYS_ID_PUBLISH` +0.053.

## 4. Multivariado

- Pearson vs TARGET: `EXT_SOURCE_3` −0.179, `_2` −0.160, `_1` −0.155 (top predictores univariados); `DAYS_BIRTH` +0.078; resto |r|<0.06. Ninguna variable explica sola (esperable en riesgo crediticio).
- Matriz de correlación entre continuas (a anexar): `AMT_CREDIT ↔ AMT_ANNUITY ↔ AMT_GOODS_PRICE` alta (>0.7, multicolinealidad a tratar en modelado); `EXT_SOURCE_*` correlación mutua moderada (~0.3–0.5, complementarias).
- Interacciones candidatas (hipótesis para modelado, no se crean aquí): `EDAD × EXT_SOURCE`, `ANUALIDAD/INGRESO × OCCUPATION`, `DPD_promedio × CREDIT_ACTIVE`.

## 5. Dimensional (granularidad y cobertura)

- `bureau`: 5.61 créditos/cliente (mediana 4, max 116). 99.45% de `SK_ID_CURR` train tienen buró.
- `bureau_balance`: 27.3M meses, media −30.7 meses historia, rango [−96,0].
- `previous_application`: ~4.93 solicitudes/cliente con historial. Approved 62.1% / Canceled 18.9% / Refused 17.4%.
- `POS`: ~29.7 meses/cliente; `installments`: ~40.1 pagos/cliente; `credit_card`: ~37.1 meses/cliente pero solo 33.68% de clientes train (103.558) tienen tarjeta → subpoblación, agregados con cobertura parcial + flag `TIENE_TARJETA`.
- Nota integridad: secundarias traen ~338–339k `SK_ID_CURR` únicos > 307.511 train porque cubren también `application_test` (fuera de alcance, se filtra por `SK_ID_CURR` train en Fase 3).

## 6. Informe de exploración (hipótesis y priorización)

1. H1: a menor `EXT_SOURCE_*`, mayor mora (r≈−0.16). Prometedoras Top-3.
2. H2: juventud + alquiler/vive con padres + baja educación + chofer/obrero = perfil de riesgo (tasas 10–17%).
3. H3: historial DPD (8.4% cuotas atrasadas, `STATUS` 1–5 en buró) discrimina mejor que montos estáticos.
4. H4: centinela empleo (18%) paga mejor (5.4% vs 8.7%) → no es desempleo, probable jubilado/estudiante; tratar como flag, no imputar.
5. Subconjuntos para Fase 3: (a) clientes sin buró (0.55%, sin historia externa), (b) con tarjeta (33.7%, señales de uso), (c) zona gris: `EXT_SOURCE` media + `DAYS_BIRTH` 25–40.
6. No se modifican objetivos de minería (clasificación desbalanceada, ROC-AUC ≥0.75).
