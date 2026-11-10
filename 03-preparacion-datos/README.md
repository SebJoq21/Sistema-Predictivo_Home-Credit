# 03 — Preparación de los Datos (CRISP-DM Fase 3) — Grupo 12

> Cubre las rúbricas **3.1 Selección**, **3.2 Limpieza**, **3.3 Construcción (Ingeniería de Características)**, **3.4 Integración Relacional** y **3.5 Formateo y Partición Anti-Leakage**.

## Artefactos y Entregables de la Fase 3

| Artefacto | Tarea Jira | Descripción |
| :--- | :---: | :--- |
| [`scripts/make_sample.py`](scripts/make_sample.py) + [`../data/sample/sample_ids.csv`](../data/sample/sample_ids.csv) | `SCRUM-17` | Muestra del 10% estratificada por `TARGET` ($30,752$ IDs, mora $8.07\%$, `random_state=42`). |
| [`scripts/load_data.py`](scripts/load_data.py) + [`../utils/memory_optimization.py`](../utils/memory_optimization.py) | `SCRUM-17` | Lectura optimizada por bloques (`chunksize=1_000_000`) y reducción de memoria (`float32`, `int8/16/32`, `category`). |
| [`scripts/clean_application.py`](scripts/clean_application.py) + [`fase3_preparacion.ipynb`](fase3_preparacion.ipynb) | `SCRUM-18`, `SCRUM-19` | Limpieza v1 de `application_train` ($307,511 \times 141$, $105.28\text{ MB}$): centinela `365243` $\rightarrow$ `FLAG_NO_EMPLEADO`, `XNA` $\rightarrow$ `NaN`, topeo `p99.9` con `TOPEADO_*`, $24$ banderas `FALTA_*` por bloque y 7 figuras en [`figuras/`](figuras/). |
| [`scripts/agg_bureau.py`](scripts/agg_bureau.py) | `SCRUM-20` | Agregación en 2 niveles (`bureau_balance` $\rightarrow$ `bureau` $\rightarrow$ `SK_ID_CURR`): `data/processed/bureau_agg.parquet` ($263,491 \times 27$). |
| [`scripts/agg_previous_pos.py`](scripts/agg_previous_pos.py) | `SCRUM-21` | Agregación de `previous_application` y `POS_CASH_balance` por `SK_ID_CURR`: `data/processed/previous_pos_agg.parquet` ($292,224 \times 30$). |
| [`scripts/agg_cards.py`](scripts/agg_cards.py) | `SCRUM-22` | Agregación de `installments_payments` y `credit_card_balance` por `SK_ID_CURR`: `data/processed/cc_installments_agg.parquet` ($291,874 \times 31$). |
| [`checklist_leakage.md`](checklist_leakage.md) | `SCRUM-23` | Verificación de las 7 reglas de control contra fuga de datos (*data leakage*). |
| [`scripts/build_dataset.py`](scripts/build_dataset.py) | `SCRUM-24`, `SCRUM-25` | Consolidación en `data/processed/dataset_consolidado.parquet` ($307,511 \times 245$) y creación de 19 variables derivadas en `data/processed/features_derivadas.parquet` + [`diccionario_features_derivadas.csv`](diccionario_features_derivadas.csv). |
| [`lista_features_final.csv`](lista_features_final.csv) | `SCRUM-26` | Selección final de **$240$ variables predictoras** (`incluir_modelo=True`) y reserva de $3$ variables protegidas para auditoría de equidad. |
| [`scripts/split_dataset.py`](scripts/split_dataset.py) | `SCRUM-27`, `SCRUM-29` | Partición estratificada 80/20 en `data/processed/train.parquet` ($246,008$ filas, mora $8.0729\%$) y `data/processed/val.parquet` ($61,503$ filas, mora $8.0728\%$). |
| [`anexo_auditoria_preliminar.md`](anexo_auditoria_preliminar.md) | `SCRUM-28` | Dictamen sobre variables sensibles (`CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `EDAD_ANIOS`) y variables proxy bajo la Ley N.° 29733 y normativa SBS. |
| [`03_informe_preparacion_datos.md`](03_informe_preparacion_datos.md) + [`03-PREPARACIÓN DE DATOS.pdf`](03-PREPARACIÓN%20DE%20DATOS.pdf) | Cierre Fase 3 | Informe completo con todas las tablas y figuras integradas. |

## Reproducibilidad del Pipeline Completo

Desde la raíz del repositorio (`Sistema-Predictivo_Home-Credit/`):

```powershell
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\make_sample.py
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\clean_application.py
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\agg_bureau.py
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\agg_previous_pos.py
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\agg_cards.py
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\build_dataset.py
.\.venv\Scripts\python.exe 03-preparacion-datos\scripts\split_dataset.py
```
