# 04 — Modelado (CRISP-DM Fase 4) — Grupo 12

> Cubre las rúbricas **4.1 Selección de técnicas de modelado**, **4.2 Diseño de pruebas (Validación Cruzada 5-Fold y Hold-Out)**, **4.3 Construcción de modelos y ajuste de parámetros**, **4.4 Evaluación técnica comparativa** y el **Andamiaje del Servicio de Inferencia (`FastAPI` + `Docker`)**.

## Artefactos y Entregables de la Fase 4

| Artefacto | Tarea Cronograma | Descripción |
| :--- | :---: | :--- |
| [`scripts/data_loader.py`](scripts/data_loader.py) | Gobernanza F3 $\rightarrow$ F4 | Carga estandarizada de `train.parquet` ($246,008 \times 240$) y `val.parquet` ($61,503 \times 240$), verificando la exclusión estricta de `CODE_GENDER`, `NAME_EDUCATION_TYPE` y `NAME_FAMILY_STATUS` (Ley N.° 29733 / SBS). |
| [`scripts/train_baseline.py`](scripts/train_baseline.py) + [`salida_fase4/metrics_baseline.json`](salida_fase4/metrics_baseline.json) | `F4.1` *(Fila 21)* | Entrenamiento del control `DummyClassifier` ($\text{Acc}=91.93\%$, $\text{ROC-AUC}=0.5000$) y del `Pipeline` anti-leakage de **Regresión Logística** (`models/modelo_baseline_logreg.pkl`, $\text{ROC-AUC}_{\text{val}}=0.7682$, $\text{PR-AUC}_{\text{val}}=0.2541$, $\text{KS}=0.4018$). |
| [`scripts/train_gbdt.py`](scripts/train_gbdt.py) + [`scripts/run_xgb_worker.py`](scripts/run_xgb_worker.py) + [`salida_fase4/metrics_gbdt.json`](salida_fase4/metrics_gbdt.json) | `F4.2` *(Fila 22)* | Estudio de sensibilidad de hiperparámetros y entrenamiento de **`LightGBM` Candidato** (`models/modelo_candidato_lgbm.pkl`, $\text{ROC-AUC}_{\text{val}}=0.7887$, $\text{PR-AUC}_{\text{val}}=0.2872$, $\text{KS}=0.4409$) y **`XGBoost` Candidato** (`models/modelo_candidato_xgb.pkl`, $\text{ROC-AUC}_{\text{val}}=0.7864$, $\text{PR-AUC}_{\text{val}}=0.2852$, $\text{KS}=0.4361$). |
| [`scripts/cross_validation.py`](scripts/cross_validation.py) + [`salida_fase4/resultados_cv.csv`](salida_fase4/resultados_cv.csv) + [`salida_fase4/comparativa_modelos.csv`](salida_fase4/comparativa_modelos.csv) | `F4.3` *(Fila 23)* | Validación cruzada estratificada de 5 pliegues (`5-Fold StratifiedKFold`) sobre `train.parquet` (`LightGBM`: $0.7809 \pm 0.0027$; `XGBoost`: $0.7805 \pm 0.0025$; `LogReg`: $0.7659 \pm 0.0023$). |
| [`arquitectura_inferencia.md`](arquitectura_inferencia.md) + [`figuras/diagrama_arquitectura.png`](figuras/diagrama_arquitectura.png) | `F4.4` *(Fila 24)* | Diseño arquitectónico del prototipo de inferencia en Python con capa REST y contenedor Docker. |
| [`../services/inference-service/main.py`](../services/inference-service/main.py) + [`../services/inference-service/requirements.txt`](../services/inference-service/requirements.txt) | `F4.5` *(Fila 25)* | Desarrollo inicial (borrador funcional) del microservicio de inferencia con `FastAPI` (`/health`, `/model/info`, `/predict`, `/predict/batch`). |
| [`../services/inference-service/Dockerfile`](../services/inference-service/Dockerfile) + [`../docker-compose.yml`](../docker-compose.yml) | `F4.6` *(Fila 26)* | Configuración reproducible de contenedor Docker (`python:3.11-slim` + `libgomp1`) y orquestación local con volumen `:ro`. |
| [`scripts/generate_figures.py`](scripts/generate_figures.py) + [`figuras/`](figuras/) | Visualización F4 | 7 gráficos de alta resolución (300 DPI): curvas ROC (`F01`), curvas PR (`F02`), curva KS (`F03`), boxplots CV 5-Fold (`F04`), importancia Top-25 (`F05`), distribución de scores (`F06`) y arquitectura (`diagrama_arquitectura.png`). |
| [`fase4_modelado.ipynb`](fase4_modelado.ipynb) | Notebook F4 | Notebook ejecutado con la demostrativa completa de los modelos, tablas comparativas, figuras y prueba local del servicio de inferencia. |
| [`04_informe_modelado.md`](04_informe_modelado.md) + [`04-MODELADO.pdf`](04-MODELADO.pdf) | Cierre Fase 4 | Informe técnico completo en formato Markdown y PDF académico. |

## Reproducibilidad del Pipeline de la Fase 4

Desde la raíz del repositorio (`Sistema-Predictivo_Home-Credit/`):

```powershell
.\.venv\Scripts\python.exe 04-modelado\scripts\train_baseline.py
.\.venv\Scripts\python.exe 04-modelado\scripts\train_gbdt.py
.\.venv\Scripts\python.exe 04-modelado\scripts\cross_validation.py
.\.venv\Scripts\python.exe 04-modelado\scripts\generate_figures.py
```

Para levantar el servicio de inferencia localmente:

```powershell
.\.venv\Scripts\uvicorn.exe services.inference-service.main:app --port 8000
# O mediante Docker Compose:
docker compose up --build
```
