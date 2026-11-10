# Checklist de Validación Anti-Leakage (`SCRUM-23`)

**Proyecto:** Sistema Predictivo de Riesgo Crediticio (Home Credit) · Grupo 12  
**Fase CRISP-DM:** Fase 3 — Preparación de los Datos  
**Responsable:** Ashley Elizabeth Bayona Vera (Colaborador: Gabriel Sonny De la Cruz Aguilar)

---

## 1. Objetivo

Garantizar que ninguna transformación, limpieza, agregación de tablas secundarias ni partición del dataset introduzca **fuga de datos (*data leakage*)** (temporal, entre clientes o entre los subconjuntos de entrenamiento y validación) que infle artificialmente el $\text{ROC-AUC}$ en la Fase 4 y Fase 5.

---

## 2. Verificaciones Ejecutadas y Evidencia Cuantitativa

| # | Regla Anti-Leakage | Estado | Evidencia Verificada en el Pipeline (`scripts/`) |
| :--- | :--- | :---: | :--- |
| **1** | **Exclusión total de `application_test.csv`** | **CUMPLIDO** | `application_test.csv` no se carga en ningún script. En `agg_bureau.py`, `agg_previous_pos.py` y `agg_cards.py` todas las tablas secundarias se filtran previamente al conjunto de `SK_ID_CURR` presentes en `application_train.csv` ($307,511$ solicitudes). Se descartaron $251,103$ filas de `bureau`, $12,598,313$ de `bureau_balance`, $256,513$ de `previous_application`, $1,457,983$ de `POS_CASH_balance`, $2,013,809$ de `installments_payments` y $612,347$ de `credit_card_balance` pertenecientes a `test`. |
| **2** | **Disyunción estricta a nivel cliente (`SK_ID_CURR`) entre `train` y `val`** | **CUMPLIDO** | En `split_dataset.py` la partición 80/20 estratificada por `TARGET` (`random_state=42`) produce `train.parquet` ($246,008$ clientes, mora $8.0729\%$) y `val.parquet` ($61,503$ clientes, mora $8.0728\%$). La intersección de llaves es exactamente $\lvert \text{SK\_ID\_CURR}_{\text{train}} \cap \text{SK\_ID\_CURR}_{\text{val}} \rvert = 0$. |
| **3** | **Agregaciones secundarias aisladas por cliente (sin cruzar filas entre clientes)** | **CUMPLIDO** | Todas las variables agregadas (`BUREAU_*`, `PREV_*`, `POS_*`, `INST_*`, `CC_*`) se calculan exclusivamente mediante `groupby("SK_ID_CURR")` (o `SK_ID_BUREAU` $\rightarrow$ `SK_ID_CURR`). Ninguna estadística de agregación utiliza información de otros solicitantes ni de la etiqueta `TARGET`. |
| **4** | **Causalidad temporal (solo historia pasada respecto a la solicitud)** | **CUMPLIDO** | Todas las variables temporales en las 7 tablas son relativas a la fecha de solicitud y cumplen $\text{DAYS\_*} \le 0$ y $\text{MONTHS\_BALANCE} \le 0$ (rango $[-96, 0]$ meses). No existen registros posteriores a la originación del crédito (`DAYS_BIRTH >= 0`: $0$ casos; `DAYS_REGISTRATION > 0`: $0$ casos; `DAYS_ID_PUBLISH > 0`: $0$ casos). |
| **5** | **Recálculo de cuantiles de topeo (`p99.9`) exclusivamente sobre `train`** | **CUMPLIDO** | En `split_dataset.py`, los umbrales del percentil $99.9$ de las 4 variables acotadas se ajustan únicamente sobre `train.parquet` (`AMT_INCOME_TOTAL = 900,000.0`, `OBS_30_CNT_SOCIAL_CIRCLE = 17.0`, `OBS_60_CNT_SOCIAL_CIRCLE = 16.0`, `AMT_REQ_CREDIT_BUREAU_QRT = 4.0`) y se aplican como transformación fija a `val.parquet`. |
| **6** | **No imputación global antes del split** | **CUMPLIDO** | Conforme a la política de no imputación, los valores faltantes se mantienen como `NaN` para LightGBM/XGBoost junto con las $24$ banderas `FALTA_*` (derivadas fila por fila sin usar estadísticas poblacionales). Para el modelo baseline de Regresión Logística (Fase 4), cualquier imputación o escalado (`SimpleImputer`, `StandardScaler`) se ejecutará dentro de un `sklearn.pipeline.Pipeline` ajustado (`fit`) únicamente sobre `train`. |
| **7** | **Ausencia de variables proxy del `TARGET` post-evento** | **CUMPLIDO** | Ninguna bandera derivada (`FLAG_NO_EMPLEADO`, `FALTA_*`, `TOPEADO_*`, `TIENE_BUREAU`, `TIENE_PREV_APP`, `TIENE_TARJETA`) ni ratio financiero utiliza la variable `TARGET` en su construcción (sin *target encoding* global). |

---

## 3. Dictamen de Cierre

El pipeline de la Fase 3 supera las **7 reglas de control anti-leakage**. Los conjuntos `data/processed/train.parquet` y `data/processed/val.parquet` quedan aprobados y certificados para iniciar el entrenamiento de modelos en la Fase 4.
