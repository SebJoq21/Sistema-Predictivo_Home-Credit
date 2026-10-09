# 03 — Informe de Calidad de Datos (PDF Sec 7.2/8.3, verificado en notebook Sec 6–10)

Fuente ejecutada: `01_EDA_application_train.ipynb` (17 celdas) + `02_EDA_tablas_secundarias.md` + `diccionario_datos_anotado.csv` (219 filas). Entorno: pandas 3.0.6 (`str` nativo + `category`), scipy 1.18.1, sklearn 1.9.1.

## 1. Integridad — Celda 6 (Sec 6)
- 307.511 filas, `SK_ID_CURR` únicos 307.511, duplicados 0. Nulos en llaves/`TARGET`: 0.
- `TARGET`: 0 = 282.686, 1 = 24.825, tasa mora 0.0807 (8.07%).
- 12 `str` pendientes de `category`: `NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, OCCUPATION_TYPE, WEEKDAY_APPR_PROCESS_START, ORGANIZATION_TYPE, FONDKAPREMONT_MODE, HOUSETYPE_MODE, WALLSMATERIAL_MODE, EMERGENCYSTATE_MODE`.
- Dtypes: `float64` 68, `int64` 39, `str` 12, `int8` 2, `int32` 1, `category` 4 (122 base + 4 derivadas `RATIO_CREDITO_INGRESO, RATIO_ANUALIDAD_INGRESO, EDAD_ANIOS, FLAG_NO_EMPLEADO`).

## 2. Financieras — Celda 7 (Pregunta negocio 1)
- `AMT_INCOME_TOTAL`: n=307.511, media 168.797, mediana 147.150, p99 472.500, max 117.000.000 (outlier extremo).
- `AMT_CREDIT`: media 599.925, mediana ~513.531, p99 1.850.000, max 4.050.000.
- `AMT_ANNUITY`: n=307.499 (12 nulos), mediana 24.903, p99 70.006. `AMT_GOODS_PRICE`: n=307.233 (278 nulos), mediana 450.000.
- Mediana ratio crédito/ingreso: pago 3.2667 vs mora 3.2531; anualidad/ingreso: 0.1623 vs 0.1693 (morosos con carga ligeramente mayor). Boxplots sin separación visual fuerte → el ratio solo no discrimina.

## 3. Categóricas vs mora — Sec 8.1/8.2/8.3/8.4 (chi2, ejecutadas)
Confirmadas:
- `NAME_CONTRACT_TYPE` (8.1): Cash 8.35% (278.232) vs Revolving 5.48% (29.279), p=1.024e-65.
- `CODE_GENDER` (8.1): M 10.14% (105.059) vs F 7.0% (202.448) vs XNA 0% (4), p=1.129e-200. Paridad F/M ≈ 0.69 < 0.8 → **hallazgo de sesgo** (XNA n=4 no concluyente).
- `FLAG_OWN_CAR` (8.2): N 8.5% (202.924) vs Y 7.24% (104.587), p=9.331e-34.
- `FLAG_OWN_REALTY` (8.2): N 8.32% (94.199) vs Y 7.96% (213.312), p=0.0006681 (diferencia leve pero significativa).
- `NAME_FAMILY_STATUS` (8.3): Civil marriage 9.94% (29.775), Single 9.81% (45.444), Separated 8.19% (19.770), Married 7.56% (196.432), Widow 5.82% (16.088), Unknown 0% (2), p=7.745e-107.
- `NAME_EDUCATION_TYPE` (8.4): Lower secondary 10.93% (3.816), Secondary 8.94% (218.391), Incomplete higher 8.48% (10.277), Higher education 5.36% (74.863), Academic degree 1.83% (164), p=2.448e-219 → gradiente educativo claro, auditoría prioritaria junto a género.
- `NAME_HOUSING_TYPE` (8.3, 6/6): Rented apartment 12.31% (4.881), With parents 11.70% (14.840), Municipal apartment 8.54% (11.183), Co-op apartment 7.93% (1.122), House/apartment 7.8% (272.868), Office apartment 6.57% (2.617), p=1.099e-88 → inquilinos y jóvenes con padres como focos de riesgo.

## 4. Temporales — Celda 9 (Sec 6.2)
- Edad: pago 44.18 (sd 11.95) vs mora 40.75 (sd 11.48) → morosos ~3.4 años más jóvenes. Histograma confirma mayor densidad mora entre 25–40.
- Centinela `DAYS_EMPLOYED==365243`: 55.374 (18.01%) → `FLAG_NO_EMPLEADO`. Mora con flag 5.40% vs sin flag 8.66% (el centinela paga mejor; hipótesis: jubilados/estudiantes, no desempleados).
- Correlación vs TARGET: `EDAD_ANIOS` −0.078, `DAYS_EMPLOYED` −0.045, `DAYS_REGISTRATION` +0.042, `DAYS_ID_PUBLISH` +0.052 (todas débiles).

## 5. Completitud y resto de dimensiones (base previa, vigente)
- 67/122 cols con nulos; vivienda 59–70% (`COMMONAREA_*` 69.9%, `NONLIVINGAPARTMENTS_*` 69.4%, `FONDKAPREMONT` 68.4%, `LIVINGAPARTMENTS_*` 68.4%, `FLOORSMIN_*` 67.9%, `YEARS_BUILD_*` 66.5%, `OWN_CAR_AGE` 66.0%, `LANDAREA_*` 59.4%); scores `EXT_SOURCE_1` 56.4%, `_3` 19.8%, `_2` 0.2% con corr −0.155/−0.179/−0.160.
- Consistencia: diccionario dice `previous_application` 38 vars, real 37 cols. Razonabilidad: nulos vivienda/vehículo = ausencia informativa.

## 6. Baseline dummy — Celda 10 (Sec 5/11)
- Siempre-0: accuracy 0.9193 (engañoso) vs ROC-AUC 0.5000. Justifica ROC-AUC ≥ 0.75 y salida < 0.70. Costos FN=S/1.800 / FP=S/150.

## Decisión vinculante a Fase 2
No imputar con media/moda; flags `FALTA_X` + LightGBM/XGBoost nativo NaN; agregar 6 secundarias por `SK_ID_CURR` a Parquet; split estratificado previo anti-leakage; auditar sensibles (género prioritario por paridad 0.69).
