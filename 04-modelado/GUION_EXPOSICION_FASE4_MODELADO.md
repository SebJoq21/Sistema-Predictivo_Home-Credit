# Guion de Exposición — Fase 4: Modelado (CRISP-DM)
## Sistema Predictivo de Riesgo Crediticio (*Home Credit Default Risk*)
### Guion Oral Verbatim de 10 Minutos — Navegación Estricta y Lineal por el Notebook

* **Institución:** Universidad Nacional Mayor de San Marcos (UNMSM)  
* **Facultad:** Facultad de Ingeniería de Sistemas e Informática (FISI)  
* **Escuela Profesional:** Ingeniería de Software  
* **Asignatura:** Minería de Datos · **Docente:** Pedro Martín Lezama Gonzales  
* **Grupo:** Grupo 12  
* **Integrantes:** Gabriel Sonny De la Cruz Aguilar · Diego Martin Arevalo Mauricio · Sebastian Joaquin Chicata Serrato · David Sebastian Piñarreta Rojas · Ashley Elizabeth Bayona Vera  
* **Notebook de Soporte:** [`04-modelado/fase4_modelado.ipynb`](./Sistema-Predictivo_Home-Credit/04-modelado/fase4_modelado.ipynb)  
* **Tiempo Límite Estricto:** **10 minutos exactos** (orden 100% descendente, sin retrocesos de celdas).

---

## Cronograma y Mapa de Scroll Lineal (De Arriba hacia Abajo)

El avance en el notebook es estrictamente descendente: **Sección 1 $\rightarrow$ 2 $\rightarrow$ 3 $\rightarrow$ 4 $\rightarrow$ 5 $\rightarrow$ 6 $\rightarrow$ 7**.

| Tiempo | Hito / Sección del Notebook | Celdas | Qué Mostrar / Enfocar en Pantalla |
| :---: | :--- | :---: | :--- |
| `00:00 - 01:00` (1 min) | **Inicio:** Portada y Objetivos de Fase 4 | **00 y 01** | Título principal del notebook, setup del Grupo 12 y rúbricas cubiertas. |
| `01:00 - 02:15` (1:15 min) | **Sección 1:** Gobernanza, Insumos F3 y Plan de Pruebas (Rúbricas 4.1 y 4.2) | **02 y 03** | Tabla con tamaños de `train` (246k), `val` (61k), tasa de mora (8.07%), exclusión de variables protegidas y matriz `TC-01` a `TC-07`. |
| `02:15 - 03:30` (1:15 min) | **Sección 2:** Selección de Técnicas, Escalera de Complejidad y Baseline (Rúbrica 4.1) | **04 y 05** | Escalera de complejidad (Niveles 0-4), casos reales (Basilea/Fintechs), papers Q1-Q3, control Dummy ($Acc=91.9\%$) y coeficientes LogReg. |
| `03:30 - 04:45` (1:15 min) | **Sección 3:** Estudio de Sensibilidad GBDT (Rúbrica 4.3) | **06 y 07** | Tabla comparativa `df_params` (`LGBM-1` vs `LGBM-3` con Recall de 4.45% vs 66.08%, y control de sobreajuste). |
| `04:45 - 06:00` (1:15 min) | **Sección 4:** 5-Fold CV, Significancia Estadística y Tabla Maestra (Rúbricas 4.2 y 4.4) | **08 y 09** | Tabla CV 5-Fold ($\mu \pm \sigma$), Test t pareado ($t=23.80, p < 0.001$) y tabla maestra comparativa (AUC 0.7887 y ahorro del 45.4%). |
| `06:00 - 07:15` (1:15 min) | **Sección 5:** Curvas de Evaluación Técnica y Calibración Brier | **10 y 11** | Gráficos: Curvas ROC (`F01`), PR (`F02`), KS (`F03`), Boxplots CV (`F04`) y Calibración Probabilística Brier (`F07`). |
| `07:15 - 08:30` (1:15 min) | **Sección 6:** Patrones Predictivos e Importancia | **12 y 13** | Tabla de Ganancia por Familia (% Gain), gráfico `F05` (Top-25) y gráfico `F06` (distribución de scores). |
| `08:30 - 09:30` (1 min) | **Sección 7:** Prototipo de Inferencia (FastAPI + Docker) | **14 y 15** | `diagrama_arquitectura.png`, JSON de prueba y segmentación tricotómica (Zona Gris al 26.44%). |
| `09:30 - 10:00` (30 s) | **Cierre:** Conclusiones y Transición a Fase 5 | **Fin Celda 15** | Cumplimiento pleno del Charter, 7 casos de prueba aprobados y pase formal a Evaluación. |

---

# GUION VERBATIM (PALABRA POR PALABRA — 10 MINUTOS)

---

### [MINUTO 00:00 - 01:00] INICIO: PORTADA Y CONTEXTO DEL NEGOCIO

> 📍 **EN EL NOTEBOOK:**  
> Ubicarse en el tope del archivo: **Celdas 00 y 01**.  
> * **Celda 00 [Markdown]:** `# Fase 4: Modelado (CRISP-DM) — Notebook de Experimentación y Validación`  
> * **Celda 01 [Código]:** Importaciones y configuración de rutas.

> **[ACCIÓN: Voz clara, pausada y mirando al jurado. Mantener el título en pantalla.]**
>
> "Buenos días, profesor Pedro Lezama y compañeros. 
> 
> A nombre del **Grupo 12**, presentamos la sustentación de la **Fase 4: Modelado**, bajo la metodología CRISP-DM, para nuestro **Sistema Predictivo de Riesgo Crediticio sobre la cartera de Home Credit**.
> 
> Como vemos en este notebook de experimentación, el objetivo de esta fase es entrenar, calibrar y validar algoritmos supervisados capaces de estimar con alta precisión la probabilidad de impago de un cliente. 
> 
> Nuestro enfoque no se limita a métricas teóricas: responde directamente a los objetivos de negocio de nuestro Project Charter: **reducir las pérdidas financieras por mora**, **automatizar la originación crediticia** y garantizar una evaluación **ética y sin sesgos**, en estricto cumplimiento con la regulación bancaria."

---

### [MINUTO 01:00 - 02:15] SECCIÓN 1: GOBERNANZA E INSUMOS CERTIFICADOS DE FASE 3

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la **Sección 1**.  
> * **Celda 02 [Markdown]:** `## 1. Verificación de Datos, Gobernanza y Plan de Pruebas (Rúbricas 4.1 y 4.2)`  
> * **Celda 03 [Código]:** Verificación de metadatos de los archivos Parquet.  
> 👁️ **Enfocar en pantalla:** La tabla con las filas de `train` ($246,008$), `val` ($61,503$), la tasa de mora del **$8.07\%$** y la exclusión de las 3 variables protegidas.

> **[ACCIÓN: Señalar la tabla de datos y las variables excluidas en pantalla.]**
>
> "Iniciamos en la **Sección 1 del notebook**, donde recibimos los conjuntos certificados de la Fase 3 en formato columnar **Parquet**, el cual elegimos porque conserva intactos los tipos de datos optimizados (`float32` y `category`) y permite una lectura en memoria ultrarrápida.
> 
> Trabajamos con dos particiones completamente disyuntas a nivel de cliente (`SK_ID_CURR`):
> * `train.parquet` con **246,008 solicitudes**.
> * `val.parquet` con **61,503 solicitudes**.
> 
> Hay cero clientes repetidos entre ambos conjuntos y ambos preservan la prevalencia real de mora del **8.07%**. Esto representa un desbalance severo: **1 cliente moroso por cada 11.4 clientes cumplidos**.
> 
> Asimismo, formalizamos nuestro **Plan de Pruebas (Rúbrica 4.2)** bajo una matriz de 7 casos de prueba (`TC-01` a `TC-07`) que audita desde la prevención de fuga de datos (*data leakage*) y la estabilidad estadística, hasta el cumplimiento estricto de la **Ley N.° 29733 de Protección de Datos Personales** y las normas de la **SBS**: congelamos **240 variables predictoras autorizadas** y excluimos del modelado el género (`CODE_GENDER`), el nivel educativo y el estado civil, reservándolas exclusivamente para la auditoría de sesgo e impacto dispar en la Fase 5."

---

### [MINUTO 02:15 - 03:30] SECCIÓN 2: SELECCIÓN DE TÉCNICAS, ESCALERA DE COMPLEJIDAD Y BASELINE (RÚBRICA 4.1)

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la **Sección 2**.  
> * **Celda 04 [Markdown]:** `## 2. Selección de Técnicas de Modelado (Rúbrica 4.1), Baseline Lineal y Control Dummy`  
> * **Celda 05 [Código]:** Métricas del Dummy y visualización de `coeficientes_logreg_top25.csv`.  
> 👁️ **Enfocar en pantalla:** La escalera de complejidad algorítmica, los papers Q1/Q2/Q3 y la tabla de coeficientes ($\beta$) de Regresión Logística.

> **[ACCIÓN: Señalar en pantalla la progresión de menor a mayor complejidad y los coeficientes en conflicto.]**
>
> "Avanzamos a la **Sección 2**, correspondiente a la Rúbrica 4.1: Selección de Técnicas.
> 
> Aquí establecemos una **progresión estricta de menor a mayor complejidad**, respaldada por **casos reales de la industria** y **literatura científica en revistas Q1, Q2 y Q3**:
> 
> 1. En el **Nivel 0**, el **DummyClassifier** como control estadístico de habilidad nula: si el banco aprobara a todos prediciendo `TARGET=0`, tendría un Accuracy engañoso del **91.93%**, pero un ROC-AUC de **0.50** y un Recall de **0%**, demostrando el peligro de evaluar clases desbalanceadas solo con exactitud.
> 2. En el **Nivel 1**, construimos nuestro **Baseline: Regresión Logística** con regularización $L_2$. Este es el estándar regulatorio tradicional exigido por **Basilea II/III** y la **SBS** en banca comercial (como BBVA o BCP) por su interpretabilidad aditiva. Lo implementamos con un `Pipeline` anti-leakage ajustado estrictamente dentro de train.
> 3. Aunque la regresión logística alcanza un ROC-AUC respetable de **0.7682**, la tabla de coeficientes evidencia su debilidad: genera signos opuestos artificiales entre variables colineales como `AMT_GOODS_PRICE` ($-0.58$) y `AMT_CREDIT` ($+0.43$).
> 4. Por ello, y sustentados en **papers Q1 canónicos** como el de *Lessmann et al. (2015 en EJOR)* y *Xia et al. (2017 en ESWA)*, así como en **casos reales de Fintechs como Home Credit y Nubank** (que redujeron entre 15% y 25% sus pérdidas brutas), escalamos hacia los **Niveles 3 y 4 de alta complejidad: XGBoost y LightGBM**. Estos algoritmos de árboles potenciados manejan de forma no lineal interacciones complejas y cuentan con **enrutamiento nativo de valores ausentes (MNAR)** sin requerir imputaciones artificiales."

---

### [MINUTO 03:30 - 04:45] SECCIÓN 3: ESTUDIO DE SENSIBILIDAD EN GBDT (RÚBRICA 4.3)

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la **Sección 3**.  
> * **Celda 06 [Markdown]:** `## 3. Estudio de Sensibilidad y Ajuste de Parámetros en GBDT (Tarea F4.2 — Rúbricas 4.3 y 4.4)`  
> * **Celda 07 [Código]:** Visualización de `df_params` (`ajuste_parametros_gbdt.csv`).  
> 👁️ **Enfocar en pantalla:** La tabla comparativa horizontal con `LGBM-1`, `LGBM-2`, `LGBM-3` y `XGB-4`.

> **[ACCIÓN: Señalar en la tabla la fila LGBM-1 vs LGBM-3 en la columna de Recall y brecha de AUC.]**
>
> "Siguiendo hacia la **Sección 3 del notebook**, abordamos la Rúbrica 4.3: Ajuste de Hiperparámetros. 
> 
> Realizamos un estudio de sensibilidad con cuatro configuraciones que nos aportó dos conclusiones clave:
> 
> 1. **El impacto de ponderar las clases (`scale_pos_weight = 11.3871`):**  
> Fíjense en la fila `LGBM-1`: al entrenar sin pesos, el modelo optimiza la mayoría y, al umbral clásico de 0.50, **solo detecta al 4.45% de los morosos**. En cambio, en `LGBM-3`, al fijar el peso en 11.3871 y monitorear el AUC en early stopping, el Recall al corte 0.50 sube de inmediato al **66.08%** y el coeficiente MCC mejora a **0.2645**.
> 2. **El control del sobreajuste:**  
> En la fila `LGBM-2`, al permitir 63 hojas sin límite de profundidad, el modelo sobreajustó en train (AUC 0.8936) pero se estancó en validación con una brecha de casi **11 puntos porcentuales**.  
> Por ello, en nuestro modelo candidato **`LGBM-3`**, aplicamos regularización estructural: limitamos `max_depth=6`, `num_leaves=31`, `min_child_samples=100` y penalizaciones $L_1=2$ y $L_2=5$.
> 
> Con esto, el modelo aprendió durante **929 árboles**, alcanzando el mejor equilibrio del proyecto: **ROC-AUC de 0.7887**, **PR-AUC de 0.2872** y el menor costo financiero esperado."

---

### [MINUTO 04:45 - 06:00] SECCIÓN 4: VALIDACIÓN CRUZADA 5-FOLD Y TABLA MAESTRA (RÚBRICAS 4.2 Y 4.4)

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la **Sección 4**.  
> * **Celda 08 [Markdown]:** `## 4. Validación Cruzada Estratificada de 5 Pliegues, Significancia Estadística y Comparativa Maestra (Tarea F4.3)`  
> * **Celda 09 [Código]:** Salidas de `df_cv_resumen` y tabla maestra `df_comp` (`comparativa_modelos.csv`).  
> 👁️ **Enfocar en pantalla:** La primera tabla con la media y desviación estándar de la CV de 5 pliegues, y la segunda tabla comparativa maestra en validación hold-out.

> **[ACCIÓN: Señalar en la primera tabla la desviación estándar baja y en la segunda tabla los costos financieros.]**
>
> "Continuamos bajando a la **Sección 4**, donde integramos el Diseño de Pruebas (Rúbrica 4.2) y la Comparativa Maestra (Rúbrica 4.4):
> 
> Aplicamos **Validación Cruzada Estratificada de 5 Pliegues (`5-Fold StratifiedKFold`)** sobre `train.parquet`. Como se observa en la primera tabla, `LightGBM` obtiene un ROC-AUC de **$0.78092 \pm 0.00268$**, certificando una varianza mínima y altísima estabilidad entre pliegues.
> 
> Para confirmar que esta ganancia es genuina, ejecutamos un **Test t pareado de Student y el Test de Wilcoxon** (`pruebas_significancia_cv.csv`): LightGBM supera a la Regresión Logística con un valor $t = 23.80$ y un **$p$-valor de $1.85 \times 10^{-5}$ ($p < 0.001$)**, ratificando significancia estadística estricta. Frente a XGBoost, ambos GBDT son estadísticamente equivalentes ($p = 0.53$), pero LightGBM se elige como candidato principal por su eficiencia computacional $3.3\times$ mayor.
> 
> Y en la segunda tabla, evaluando el conjunto independiente de validación hold-out de 61,503 solicitudes:
> * La **Regresión Logística** logra un ROC-AUC de **0.7682** y KS de **0.4018**.
> * **XGBoost** logra un desempeño competitivo muy alto con ROC-AUC de **0.7864** y PR-AUC de **0.2852**.
> * Pero **LightGBM se consolida como el modelo ganador**:
>   * **ROC-AUC en Validación:** **0.78865**, superando la meta de 0.75 del Project Charter.
>   * **PR-AUC:** **0.28720**, superando en más de 3.5 veces el azar.
>   * **Estadístico KS:** **0.44086**, superando el 0.40 de exigencia bancaria.
>   * **Impacto Económico:** Modelando un costo de S/ 1,800 por Falso Negativo y S/ 150 por Falso Positivo, LightGBM en su umbral óptimo ($\tau^* = 0.445$) reduce el costo esperado de **S/ 145.31 a S/ 79.33 por solicitud**, lo que significa un **ahorro financiero del 45.41%** capturando al **72.89% de los morosos reales**."

---

### [MINUTO 06:00 - 07:15] SECCIÓN 5: CURVAS DE EVALUACIÓN TÉCNICA (ROC, PR, KS Y BOXPLOTS)

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la **Sección 5**.  
> * **Celda 10 [Markdown]:** `## 5. Curvas de Evaluación Técnica y Calibración (ROC, Precision-Recall, KS, Boxplots CV y Brier Score)`  
> * **Celda 11 [Código]:** Despliegue de figuras `F01`, `F02`, `F03`, `F04` y `F07`.  
> 👁️ **Enfocar en pantalla:** Las curvas comparativas ROC (`F01`), Precision-Recall (`F02`), curva KS (`F03`) y curva de calibración (`F07`).

> **[ACCIÓN: Señalar la curva ROC, la curva KS y finalmente la curva de calibración F07.]**
>
> "En la **Sección 5 del notebook** respaldamos visualmente esta evaluación técnica mediante gráficos de alta resolución:
> 
> * En la **Figura F01 (Curva ROC)**, observamos cómo las curvas de LightGBM (verde) y XGBoost (naranja) dominan a lo largo de todos los umbrales operativos sobre la Regresión Logística (azul) y el Dummy (línea discontinua).
> * En la **Figura F02 (Curva Precision-Recall)**, confirmamos que para cualquier nivel de cobertura de morosos, los árboles gradient boosting sostienen una precisión muy superior a la prevalencia base del 8.07%.
> * En la **Figura F03 (Curva Kolmogorov-Smirnov)**, vemos la máxima separación entre la distribución acumulada de morosos y cumplidos, alcanzando el pico de **$KS = 0.4409$**.
> * En la **Figura F04**, los boxplots de la validación cruzada evidencian distribuciones compactas que ratifican la consistencia analítica de los resultados.
> * Y en la **Figura F07 (Calibración de Probabilidades y Brier Score)**, mostramos que los scores operativos con pesos son excelentes para ranking y zona gris, mientras que al aplicar la calibración bayesiana de Saerens-Elkan para provisiones bancarias (SBS/Basilea), LightGBM alcanza un **Brier Score sobresaliente de $0.0659$**, batiendo al Dummy ($0.0807$) y a la Regresión Logística ($0.0672$)."

---

### [MINUTO 07:15 - 08:30] SECCIÓN 6: PATRONES PREDICTIVOS Y FEATURE IMPORTANCE POR FAMILIA

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la **Sección 6**.  
> * **Celda 12 [Markdown]:** `## 6. Patrones Predictivos Extraídos e Importancia de Variables por Familia Relacional`  
> * **Celda 13 [Código]:** Tabla de familias relacionales, figura `F05` (Top-25) y figura `F06` (distribución de scores).  
> 👁️ **Enfocar en pantalla:** La tabla de ganancia (% Gain) por familia y el gráfico de barras horizontales de la Figura F05.

> **[ACCIÓN: Señalar la tabla de porcentajes por familia y la variable líder EXT_SOURCE_MEAN.]**
>
> "Siguiendo hacia la **Sección 6 del notebook**, extrajimos la ganancia de información (*Gain*) de las 27,525 particiones realizadas en los 929 árboles de LightGBM.
> 
> Esta descomposición valida plenamente la ingeniería de características relacionales de la Fase 3:
> 
> * El **31.45% de la ganancia total** proviene de los **Scores Externos**, liderados por nuestra variable sintética `EXT_SOURCE_MEAN`, que ocupa el puesto 1 absoluto aportando por sí sola el **17.70% del poder predictivo**.
> * El **20.77%** proviene de variables **Demográficas y de Estabilidad Laboral**, destacando `ORGANIZATION_TYPE` (8.58%) y `OCCUPATION_TYPE` (2.43%). El modelo aprende el riesgo crediticio a través de la estabilidad del sector económico del cliente.
> * Y **más del 33% del poder predictivo proviene directamente de las tablas secundarias integradas**: buró de deudas previas (`BUREAU_DEBT_RATIO_MEAN`), solicitudes previas en POS-Cash y el ratio de atraso en cuotas (`INST_LATE_RATIO`).
> 
> Y como muestra la **Figura F06** al final de la celda, las probabilidades predichas separan con claridad a la mayoría cumplidora a la izquierda y a los morosos hacia la derecha."

---

### [MINUTO 08:30 - 09:30] SECCIÓN 7: ANDAMIAJE DE INFERENCIA ANALÍTICA (FASTAPI + DOCKER)

> 📍 **EN EL NOTEBOOK:**  
> Scroll descendente hacia la última sección: **Sección 7**.  
> * **Celda 14 [Markdown]:** `## 7. Verificación del Prototipo del Servicio de Inferencia (FastAPI + Docker — Tareas F4.4 a F4.6)`  
> * **Celda 15 [Código]:** `diagrama_arquitectura.png` y salidas de prueba de los endpoints en JSON.  
> 👁️ **Enfocar en pantalla:** El diagrama de arquitectura del contenedor y las respuestas JSON de `/health` y `/predict`.

> **[ACCIÓN: Señalar el diagrama de arquitectura y la segmentación tricotómica en el JSON.]**
>
> "Llegamos a la **Sección 7 del notebook**, donde cumplimos con las Tareas F4.4, F4.5 y F4.6 de nuestro cronograma construyendo el **andamiaje operacional de inferencia analítica**:
> 
> * **Microservicio REST con FastAPI:** Expone las rutas `/health`, `/model/info`, `/predict` individual y `/predict/batch`. Valida los datos con `Pydantic v2`, bloquea atributos sensibles y responde con una latencia inferior a **15 milisegundos**.
> * **Regla de Decisión Tricotómica Preliminar:** Operativiza las predicciones en tres bandejas:
>   1. **Aprobación Automática ($p < 0.35$):** Clientes de mínimo riesgo.
>   2. **Zona Gris o Revisión Manual ($0.35 \le p \le 0.60$):** Aísla con exactitud el **26.44%** de las solicitudes para que un analista humano pida sustento adicional, cumpliendo la meta del Business Case de **automatizar con alta confianza más del 73.5% de la originación crediticia**.
>   3. **Rechazo Automático ($p > 0.60$):** Clientes de alto riesgo.
> * **Contenerización Docker Reproducible:** Construido sobre `python:3.11-slim` con usuario sin privilegios (`appuser`) y montando los modelos en un volumen de solo lectura (`:ro`)."

---

### [MINUTO 09:30 - 10:00] CIERRE: CONCLUSIONES Y TRANSICIÓN A LA FASE 5

> 📍 **EN EL NOTEBOOK:**  
> Mantenerse al pie de la Celda 15.

> **[ACCIÓN: Mirar fijamente al jurado con tono de cierre seguro y concluyente.]**
>
> "Para concluir, la Fase 4 cierra con **éxito total**:
> 1. Nuestro modelo ganador **`LightGBM` supera holgadamente todas las metas del Project Charter** con un **ROC-AUC de 0.7887**, **PR-AUC de 0.2872**, **KS de 0.4409** y una **reducción de pérdidas financieras del 45.41%**.
> 2. Elegimos formalmente a LightGBM sobre XGBoost porque ofrece menor costo financiero, entrena **3.3 veces más rápido** (74 s vs 242 s) y maneja datos categóricos nativamente.
> 3. Dejamos el microservicio de inferencia contenedorizado y listo para producción.
> 
> Con esto, quedamos plenamente listos para la **Fase 5: Evaluación**, donde realizaremos la optimización matemática de umbrales, la **auditoría regulatoria de sesgo e impacto dispar ($DIR \ge 0.80$)**, la explicabilidad local con **TreeSHAP** y la redacción del **Model Card**.
> 
> Muchas gracias. Quedamos atentos a sus preguntas."

---

## Respuestas Rápidas para la Ronda de Preguntas (30 Segundos c/u)

### P1: *"¿Por qué el Dummy tiene 91.9% de Accuracy y LightGBM 76%? ¿No es mejor el Dummy?"*
> *"Profesor, esa es la **paradoja de la exactitud**. Como el 91.93% de las solicitudes son de clientes que pagan, predecir ciegamente que todos pagan acierta el 91.9%, pero tiene Recall de 0%: no detectaría a ningún moroso y el banco quebraría. LightGBM optimiza la discriminación: detecta al 72.89% de los morosos reales con ROC-AUC de 0.7887 y ahorra más del 45% en costos."*

### P2: *"¿Cómo evitaron Data Leakage en la Regresión Logística?"*
> *"Mediante un `Pipeline` de scikit-learn. El cálculo de la imputación por mediana y el escalamiento estándar se ajustaron (`.fit`) **únicamente dentro de los pliegues de entrenamiento** de la validación cruzada. El conjunto de validación nunca participó en esos cálculos; solo recibió la transformación (`.transform`)."*

### P3: *"¿Por qué no imputaron valores nulos en LightGBM y XGBoost?"*
> *"Porque en Fase 3 demostramos con pruebas chi-cuadrado que la ausencia de datos no es aleatoria (es MNAR). No tener registro en buró o vivienda es una señal de riesgo en sí misma. Los árboles de decisión tienen **enrutamiento nativo de nulos**: evalúan en cada partición hacia qué rama maximiza la ganancia. Imputar con la media o mediana hubiera destruido esa información."*

### P4: *"¿Cómo aseguran que el modelo cumpla la Ley 29733 de Protección de Datos?"*
> *"Por diseño: `CODE_GENDER`, `NAME_EDUCATION_TYPE` y `NAME_FAMILY_STATUS` fueron excluidas físicamente de la matriz de entrenamiento de 240 columnas. En el servicio FastAPI, el validador Pydantic las descarta si un canal intenta enviarlas. Esas columnas solo se reservaron para verificar en la Fase 5 que el Disparate Impact Ratio supere el 0.80 regulatorio."*

### P5: *"¿Por qué los umbrales de 0.35 y 0.60 para la Zona Gris?"*
> *"Porque responden al requerimiento del Business Case: con $p < 0.35$ aprobamos clientes de bajo riesgo, con $p > 0.60$ rechazamos de forma directa, y la franja intermedia (0.35 a 0.60) aísla con precisión el **26.44%** de las solicitudes para evaluación manual por un analista, cumpliendo la meta de **automatizar con seguridad el 73.56%** de la originación crediticia."*

### P6: *"¿Cómo demuestran formalmente que LightGBM es estadísticamente superior a la Regresión Logística y no fue producto del azar o de una partición favorable?"*
> *"Profesor, lo certificamos mediante un **Test t pareado de Student sobre los 5 pliegues disyuntos de la validación cruzada estratificada** (`pruebas_significancia_cv.csv`). En los 5 pliegues evaluados, LightGBM superó a la Regresión Logística con una ventaja media de $+0.0150$ en ROC-AUC, arrojando un estadístico $t = 23.8046$ y un valor **$p = 1.85 \times 10^{-5}$ ($p < 0.001$)**, respaldado por la prueba no paramétrica de rangos con signo de Wilcoxon ($p = 0.0625$). Frente a XGBoost, ambos algoritmos GBDT son estadísticamente equivalentes en ranking ($p = 0.5295$), por lo que la elección final de LightGBM se sustentó en su tiempo de cómputo **$3.3\times$ más veloz** (74 s vs 242 s) y su menor costo financiero por solicitud (S/ 79.33 vs S/ 79.75)."*

### P7: *"¿Cómo calibraron las probabilidades para cumplir con el cálculo de provisiones bancarias bajo Basilea y la SBS si entrenaron con `scale_pos_weight = 11.3871`?"*
> *"Excelente pregunta. Al aplicar el ponderador de clase $w = 11.3871$, los scores se desplazan hacia la derecha para maximizar la detección de morosos y el Recall operativo ($66.08\%$), generando un Brier Score aparente de $0.1633$. Para el cálculo de Pérdida Esperada y Provisiones de capital ($PE = PD \times LGD \times EAD$), aplicamos la **transformación bayesiana post-hoc de Saerens-Elkan**: $p_{\text{cal}} = \frac{p / w}{1 - p + (p / w)}$. Esta fórmula reajusta analíticamente las probabilidades a la prevalencia poblacional real del $8.07\%$, logrando un **Brier Score sobresaliente de $0.06596$**, que bate tanto al modelo Dummy ($0.08073$) como a la Regresión Logística ($0.06721$)."*

### P8: *"¿En qué consiste su Plan de Pruebas formal (TC-01 a TC-07) y cuáles fueron los resultados obtenidos?"*
> *"Estructuramos un plan formal de 7 casos de prueba predefinidos con criterios de aceptación cuantitativos:
> 1. **TC-01 (Aislamiento muestral):** Cero clientes repetidos entre `train` y `val` $\rightarrow$ **PASS (0% fuga)**.
> 2. **TC-02 (Superación de Baselines):** ROC-AUC $> 0.75$, batiendo a Dummy y LogReg $\rightarrow$ **PASS (0.7887)**.
> 3. **TC-03 (Sensibilidad de hiperparámetros):** Recall $> 65\%$ con pesos de clase $\rightarrow$ **PASS (66.08%)**.
> 4. **TC-04 (Estabilidad en CV 5-Fold):** Desviación estándar $\sigma < 0.005$ $\rightarrow$ **PASS ($\sigma = 0.00268$)**.
> 5. **TC-05 (Significancia estadística):** Test t pareado con $p < 0.01$ $\rightarrow$ **PASS ($p = 1.85 \times 10^{-5}$)**.
> 6. **TC-06 (Cumplimiento de Fair Lending):** 3 atributos protegidos bloqueados $\rightarrow$ **PASS (100% excluidos)**.
> 7. **TC-07 (Latencia del servicio Inferencia):** Respuesta $< 50\text{ ms}$ en FastAPI $\rightarrow$ **PASS ($< 15\text{ ms}$)**.  
> El resultado global fue **7 de 7 casos APROBADOS (100% de éxito)**."*

---

## Chuleta de Métricas Clave para el Expositor (Memorización Rápida)

| Métrica / Concepto | Valor Exacto | Justificación Breve |
| :--- | :---: | :--- |
| **ROC-AUC en Validación** | **$0.7887$** | Supera la meta de $0.75$ del Project Charter y a LogReg ($0.7682$). |
| **PR-AUC en Validación** | **$0.2872$** | Más de $3.5\times$ superior a la prevalencia base del azar ($0.0807$). |
| **Estadístico Kolmogorov-Smirnov (KS)** | **$0.4409$** | Máxima separación acumulada (estándar bancario exige $> 0.40$). |
| **Ahorro Financiero por Solicitud** | **$45.41\%$** | Reduce el costo esperado de S/ 145.31 a S/ 79.33 al umbral óptimo $\tau^* = 0.445$. |
| **Recall al Umbral Óptimo** | **$72.89\%$** | Detecta a casi 3 de cada 4 morosos reales en la cartera. |
| **Estabilidad CV 5-Fold (ROC-AUC)** | **$0.7809 \pm 0.0027$** | Dispersión ínfima entre los 5 pliegues disyuntos. |
| **Test t Pareado (LGBM vs LogReg)** | **$t = 23.80, p = 1.85 \times 10^{-5}$** | Significancia estadística estricta ($p < 0.001$, LightGBM gana 5/5 pliegues). |
| **Brier Score Calibrado (Saerens-Elkan)** | **$0.06596$** | Calibración óptima para provisiones SBS/Basilea (Dummy: $0.0807$, LogReg: $0.0672$). |
| **Cobertura Zona Gris ($0.35 \le p \le 0.60$)** | **$26.44\%$** | Permite automatizar con alta seguridad el **$73.56\%$** de la originación crediticia. |
| **Velocidad de Entrenamiento GBDT** | **$74.1\text{ s vs } 242.1\text{ s}$** | LightGBM es **$3.3\times$ más rápido** que XGBoost. |
| **Plan de Pruebas (`TC-01` a `TC-07`)** | **7 / 7 PASS** | 100% de casos de prueba aprobados formalmente. |
| **Latencia Inferencia API** | **$< 15\text{ ms}$** | Servicio REST FastAPI con Pydantic v2 en contenedor Docker. |
