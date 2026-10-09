# 01 — Comprensión del Negocio y de los Datos (CRISP-DM Fase 1)

Repo técnico. Referencia teórica: `DOCUMENTOS/Compresion de Negocio - Mineria de datos.pdf` (12 secciones) + `CLASES/Comprensión del Negocio.pdf` + `Comprensión de los datos.pdf`.
Cronograma vigente: `DOCUMENTOS/Cronograma_CRISP-DM_Grupo_12.xlsx` (Fase 1: 15–22/09/2026).

## Los 8 CSV (`../data/raw/`, ~2.53 GB en disco)

| # | Archivo | Rol | Dimensión verificada |
|---|---|---|---|
| 1 | `application_train.csv` | Tabla principal. `SK_ID_CURR` + `TARGET` (1=mora, 0=pago). **307.511 filas x 122 cols**, 158.4 MB. Tasa mora **8.07%** (24.825 / 307.511). | Prioridad Fase 1 |
| 2 | `bureau.csv` | Créditos en otros burós. Llaves `SK_ID_CURR, SK_ID_BUREAU`. **1.716.428 x 17**, 162.1 MB | Secundaria |
| 3 | `bureau_balance.csv` | Historial mensual por buró. Llave `SK_ID_BUREAU`. **27.299.925 x 3**, 358.2 MB | Secundaria pesada |
| 4 | `previous_application.csv` | Solicitudes previas Home Credit. Llaves `SK_ID_CURR, SK_ID_PREV`. **1.670.214 x 37** (diccionario dice 38), 386.2 MB | Secundaria |
| 5 | `POS_CASH_balance.csv` | Estado mensual POS/cash. Llaves `SK_ID_CURR, SK_ID_PREV`. **10.001.358 x 8**, 374.5 MB | Secundaria |
| 6 | `installments_payments.csv` | Pagos en cuotas. Llaves `SK_ID_CURR, SK_ID_PREV`. **13.605.401 x 8**, 689.6 MB | Secundaria más pesada |
| 7 | `credit_card_balance.csv` | Balance tarjetas. Llaves `SK_ID_CURR, SK_ID_PREV`. **3.840.312 x 23**, 404.9 MB | Secundaria |
| 8 | `HomeCredit_columns_description.csv` | Diccionario. **219 filas**: 122 application + 17 bureau + 3 bureau_balance + 8 POS + 23 credit_card + 38 previous + 8 installments | Insumo |

Fuera de alcance: `application_test.csv` (Project Charter, límites).

## Contenido de esta carpeta

| Archivo | Cubre PDF Sec. | Entregable cronograma |
|---|---|---|
| `01_EDA_application_train.ipynb` (23 celdas: 1-5 base + 6-10 nuevas; Sec 8 en 8.1/8.2/8.3/8.4) | 6.2, 7.2, 10 | `eda_inicial.ipynb` |
| `00_alineamiento-negocio.md` | 1, 2, 3, 4 | trazabilidad a `Compresion_de_Negocio.docx` |
| `diccionario_datos_anotado.csv` (219 filas) | 6.2 | `diccionario_datos_anotado.xlsx` |
| `lista_variables_sensibles.csv` | 7.1, 11.1 | `lista_variables_sensibles.xlsx` |
| `02_EDA_tablas_secundarias.md` | 6.2 | base Fase 2 (agregación) |
| `03_informe-calidad-datos.md` | 7.2, 8.3 | justifica no-imputación |
| `04_criterios-exito.md` | 5, 10, 11 | ROC/PR-AUC/KS/MCC + umbral costo |
| `05_riesgos-supuestos-restricciones.md` | 6, 7, 8 | RAM, leakage, Ley 29733/SBS |
| `06_checklist-cierre-fase1.md` + `reporte_eda_inicial.pdf` (export) | 12 | cierre 22/09 |

## Reglas Fase 1 (no negociables)

1. No joins ni imputación aquí — solo perfilado. Joins/imputación van en Fase 2/3 con split previo anti-leakage.
2. Faltantes vivienda ~60–70% = ausencia informativa → flag binario + LightGBM/XGBoost nativo NaN.
3. Métrica: ROC-AUC ≥ 0.75 (salida < 0.70). Accuracy inválido por desbalance 91.9/8.1.
4. Despliegue: prototipo Python + capa REST + Docker (genérico). Sin Express (corregido en cronograma v3).

## Entorno verificado (.venv)
Python 3.13.5 + pandas 3.0.6 + scipy 1.18.1 + scikit-learn 1.9.1. Estándar repo: dtype `str` nativo + `category`; usar `select_dtypes(include=['str','object','string'])`. Desde Celda 8 las 7 categóricas se analizan como `category` vía variable temporal (sin mutar `df_train`; la conversión definitiva va en Fase 2).
