# Anexo de Auditoría Preliminar de Variables Sensibles y Proxies (`SCRUM-28`)

**Proyecto:** Sistema Predictivo de Riesgo Crediticio (Home Credit) · Grupo 12  
**Fase CRISP-DM:** Fase 3 — Preparación de los Datos (Ingeniería de Características y Gobernanza)  
**Marco Normativo:** Ley N.° 29733 (Protección de Datos Personales - Perú) y Normativa SBS (Gestión de Riesgo de Crédito y Equidad Algorítmica)  
**Responsable:** Ashley Elizabeth Bayona Vera (Colaborador: Gabriel Sonny De la Cruz Aguilar)

---

## 1. Propósito

Documentar la decisión técnica y regulatoria tomada en la Fase 3 sobre las **variables demográficas sensibles** y sus posibles **variables proxy**, resolviendo la *"Decisión abierta"* planteada en la Sección 2.3 del informe de Preparación de Datos y dejando trazada la estructura para la auditoría de paridad ($\ge 0.80$) en la Fase 5.

---

## 2. Evaluación de Variables Sensibles Identificadas

A partir del inventario de `01-comprension-negocio-datos/lista_variables_sensibles.csv` y las pruebas $\chi^2$ / correlaciones ejecutadas sobre las $307,511$ solicitudes de `application_train`, se clasifican los atributos sensibles en tres categorías de tratamiento:

| Variable | Categoría Sensible | Hallazgo Estadístico en `application_train` | Riesgo de Sesgo / Paridad | Decisión en Fase 3 (`lista_features_final.csv`) |
| :--- | :--- | :--- | :--- | :--- |
| **`CODE_GENDER`** | Género (Atributo protegido directo) | Tasa de mora en hombres (`M`): **$10.14\%$** ($n=105,059$) vs. mujeres (`F`): **$7.00\%$** ($n=202,448$). $\chi^2 p = 1.1 \times 10^{-200}$, $\lvert r \rvert = 0.0547$. | **Alto:** Razón de tasas de mora $F/M = 0.69 < 0.80$ (incumple regla de cuatro quintos si se usa directamente para discriminar). | **Excluida como predictora (`incluir_modelo = False`)**. Se conserva en `train.parquet` / `val.parquet` con rol `auditoria_sesgos` únicamente para auditar en Fase 5 que las predicciones del modelo mantengan paridad de aprobación $\ge 0.80$. |
| **`NAME_EDUCATION_TYPE`** | Nivel educativo (Socioeconómico protegido) | Gradiente monótono pronunciado: *Lower secondary* ($10.93\%$) $\rightarrow$ *Secondary* ($8.94\%$) $\rightarrow$ *Incomplete higher* ($8.48\%$) $\rightarrow$ *Higher education* ($5.36\%$) $\rightarrow$ *Academic degree* ($1.83\%$). $\chi^2 p = 2.4 \times 10^{-219}$. | **Medio-Alto:** Penaliza estructuralmente a solicitantes sin estudios superiores aun cuando tengan buen comportamiento transaccional. | **Excluida como predictora (`incluir_modelo = False`)**. Se conserva con rol `auditoria_sesgos` para verificar que las variables transaccionales (`INST_*`, `POS_*`, `BUREAU_*`) reemplacen el prejuicio educativo por evidencia de pago real. |
| **`NAME_FAMILY_STATUS`** | Estado civil (Atributo personal protegido) | *Civil marriage* ($9.94\%$) y *Single* ($9.81\%$) frente a *Married* ($7.56\%$) y *Widow* ($5.82\%$). $\chi^2 p = 7.7 \times 10^{-107}$, $\lvert r \rvert = 0.0041$. | **Bajo-Medio:** Señal predictiva débil ($\lvert r \rvert < 0.005$) y riesgo de discriminación por estado civil. | **Excluida como predictora (`incluir_modelo = False`)**. Se conserva con rol `auditoria_sesgos` para evaluación de impacto dispar en Fase 5. |
| **`DAYS_BIRTH` / `EDAD_ANIOS`** | Edad (Demográfica regulada con justificación actuarial) | Morosos son en promedio $\approx 3.4$ años más jóvenes ($\lvert r \rvert = 0.0782$). Rango válido verificado: $[20.5, 69.1]$ años ($0$ menores de edad). | **Controlado:** En riesgo crediticio SBS la edad refleja madurez del ciclo de ingresos/ahorro, pero debe monitorearse el tramo joven ($20\text{--}30$ años). | **Incluida con monitoreo (`incluir_modelo = True`)**. Se auditará en Fase 5 la razón de aprobación del segmento joven ($\le 30$ años) vs. adulto ($> 30$ años). |
| **`REGION_RATING_CLIENT` / `_W_CITY`** | Ubicación geográfica (Riesgo de *redlining*) | Tasa de mora por calificación regional: Rating 1 ($4.82\%$) $\rightarrow$ Rating 2 ($7.89\%$) $\rightarrow$ Rating 3 ($11.10\%$). $\lvert r \rvert = 0.0609$. | **Medio (*Proxy* socioeconómico):** Agrega riesgo macroeconómico regional sin identificar dirección ni barrio exacto (cumple Ley 29733). | **Incluida con monitoreo (`incluir_modelo = True`)**. Se verificará en Fase 5 con SHAP que su contribución no supere a los indicadores individuales de comportamiento de pago. |

---

## 3. Análisis de Variables Proxy Detectadas

1. **`OCCUPATION_TYPE` y `ORGANIZATION_TYPE` como proxies de género y educación:**
   * Ocupaciones de baja calificación (*Low-skill laborers* $17.15\%$, *Drivers* $11.33\%$) presentan mayor proporción masculina y menor nivel educativo formal, mientras que *Accountants* ($4.83\%$) y *Core staff* ($6.30\%$) concentran mayor presencia femenina y educación superior.
   * Además, la ausencia de `OCCUPATION_TYPE` ($31.35\%$ de nulos, capturada por `FALTA_OCCUPATION_TYPE`) presenta una tasa de mora menor ($6.51\%$ cuando falta vs. $8.79\%$ cuando existe, $\Delta = -2.27\text{ pp}$).
   * **Mitigación:** Al retirar `CODE_GENDER` y `NAME_EDUCATION_TYPE` del conjunto de variables predictoras y añadir $88$ variables de comportamiento transaccional real (`BUREAU_*`, `PREV_*`, `POS_*`, `INST_*`, `CC_*`) más los ratios financieros derivados, el árbol de decisión prioriza el historial de cumplimiento efectivo sobre el perfil demográfico.
2. **`FLAG_NO_EMPLEADO` (centinela `DAYS_EMPLOYED = 365243`, $18.01\%$ de la base):**
   * Corresponde en un $99.96\%$ a pensionistas/jubilados (`NAME_INCOME_TYPE = Pensioner`, $55,352$ casos) y solo $22$ casos a `Unemployed`. Su tasa de mora es **menor** que el resto de la cartera ($5.40\%$ vs. $8.66\%$), por lo que la bandera `FLAG_NO_EMPLEADO` actúa como factor protector para adultos mayores jubilados y evita penalizarlos como "sin historial laboral".

---

## 4. Protocolo para la Fase 5 (Auditoría de Equidad y Explicabilidad SHAP)

En la Fase 5 (`SCRUM-39` y `SCRUM-40`), sobre el conjunto `val.parquet` ($61,503$ solicitudes):
1. Se entrenará el modelo utilizando únicamente las **$240$ variables predictoras autorizadas** (`incluir_modelo == True` en `lista_features_final.csv`).
2. Una vez emitido el score de probabilidad y aplicado el umbral óptimo por costo esperado, se cruzará la decisión con las columnas reservadas `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS` y tramos de `EDAD_ANIOS`.
3. Se verificará que la razón de tasas de aprobación entre el grupo protegido y el grupo de referencia cumpla $\text{Disparate Impact Ratio} \ge 0.80$, ajustando los límites de la **zona gris de revisión humana** en caso de detectarse desviaciones.
