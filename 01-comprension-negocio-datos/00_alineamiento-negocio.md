# 00 — Alineamiento de Negocio (PDF Sec 1–4, resumen operativo)

Fuente: `DOCUMENTOS/Compresion de Negocio - Mineria de datos.pdf`. Esto no reemplaza el documento, lo operativiza para el repo.

## 1. Empresa (Sec 1)
- Board: CEO Radek Pluhař, CRO Martin Dvořák (sponsor, dueño de la pérdida por mora), CDO Elena Volkova + CFO (steering committee).
- Data mining = equipo Grupo 12 (Sebastián Chicata líder técnico). Riesgos valida, TI optimiza.
- Unidades afectadas: Underwriting (recibe score automático), Cobranzas (cartera más sana), Atención al cliente (respuesta casi instantánea).

## 2. Área problemática (Sec 2)
- Gestión de riesgos / Underwriting. Población no bancarizada sin historial clásico.
- Doble costo: rechazo injusto de solventes (pérdida negocio/exclusión) + aprobación de insolventes (mora 8.07% verificada: 24.825/307.511).
- Requisito: consolidar fuentes alternativas (transaccional, cuotas, buró, telecom/vivienda) en esquema relacional de 7 tablas + diccionario. Sin compra externa: `EXT_SOURCE_1/2/3` ya vienen importadas.
- Estado: aprobado por Riesgos como estratégico; hay que "venderlo" a Comercial/Atención como automatización, no como filtro restrictivo.

## 3. Solución actual (Sec 3)
- Reglas rígidas + revisión manual (umbral ingresos, cruce buró si existe, estado laboral → si no hay historial, manual).
- Pros: explicable, auditable, familiar. Contras: no escala a cientos de miles, pierde interacciones no lineales (ej. antigüedad empleo × vivienda × historial), falsos negativos y mora sostenida ~8%.

## 4. Objetivos de negocio → preguntas minería (Sec 4)
Problema minería: clasificación binaria desbalanceada sobre `TARGET`.

| Pregunta negocio (Sec 4.2) | Traducción técnica | Tabla(s) que la responde |
|---|---|---|
| ¿P(impago) por solicitante? | `P(TARGET=1 \| features)` + umbral por costo esperado (FN=S/1.800, FP=S/150) | `application_train` + agregados 6 secundarias |
| ¿Qué variables alternativas predicen buen pagador? | Importancia/SHAP Top-5; hipótesis: `EXT_SOURCE_3/2/1` > cuotas > buró | `EXT_SOURCE_*`, `installments`, `bureau` |
| ¿Qué rechazados hoy son sanos? | Segmento predicho sano pero rechazado por regla → inclusión segura | `previous_application` (REFUSED) vs `TARGET` |

Requisitos no funcionales (Sec 4.3): scoring en segundos, white-box (Top 3–5 factores por denegación), sin sesgo (paridad ≥ 0.8).
Beneficios (Sec 4.4 + Business Case): −7% pérdidas (≈S/50k/año sobre base S/720k), +aprobaciones seguras, 75% auto / 25–30% zona gris manual.
