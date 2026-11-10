"""
3.6 Consolidación de las 6 tablas agregadas por `SK_ID_CURR`, creación de variables
derivadas y selección final de variables (`SCRUM-24`, `SCRUM-25`, `SCRUM-26`).

Salidas generadas:
- `data/processed/dataset_consolidado.parquet`
- `data/processed/features_derivadas.parquet`
- `03-preparacion-datos/diccionario_features_derivadas.csv`
- `03-preparacion-datos/lista_features_final.csv`
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.memory_optimization import memory_mb, reduce_mem_usage

PROCESSED_DIR = Path("data") / "processed"
FASE3_DIR = Path("03-preparacion-datos")

# Variables protegidas por Ley 29733 / Normativa SBS (se conservan como metadato para
# la auditoría de sesgos de la Fase 5, pero se excluyen de la matriz X predictora)
PROTECTED_AUDIT_VARS = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
]


def add_derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Construye las variables derivadas de negocio documentadas en el Cronograma v3:
    1. Banderas de presencia de subpoblación (`TIENE_BUREAU`, `TIENE_PREV`, `TIENE_TARJETA`).
    2. Ratios financieros (`CREDIT_TO_INCOME_RATIO`, `ANNUITY_TO_INCOME_RATIO`,
       `ANNUITY_TO_CREDIT_RATIO`, `CREDIT_TO_GOODS_RATIO`, `INCOME_PER_FAM_MEMBER`).
    3. Variables demográficas/temporales (`EDAD_ANIOS`, `ANIOS_EMPLEO`, `EMPLEO_A_EDAD_RATIO`).
    4. Combinaciones de scores externos (`EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`,
       `EXT_SOURCE_PROD`, `EXT_SOURCE_WEIGHTED`).
    5. Indicadores compuestos de mora histórica (`HIST_DPD_TOTAL_MONTHS`, `HIST_HAS_ANY_DPD`).
    """
    dicc = []

    # 1. Banderas de presencia relacional (evita confundir ausencia de historial con 0)
    df["TIENE_BUREAU"] = df["BUREAU_CREDIT_COUNT"].notna().astype("int8")
    dicc.append({"feature": "TIENE_BUREAU", "grupo": "Presencia relacional", "formula": "1 si el cliente registra créditos en bureau.csv, 0 si no tiene buró"})

    df["TIENE_PREV_APP"] = df["PREV_APP_COUNT"].notna().astype("int8")
    dicc.append({"feature": "TIENE_PREV_APP", "grupo": "Presencia relacional", "formula": "1 si el cliente tiene solicitudes previas en Home Credit, 0 en caso contrario"})

    df["TIENE_TARJETA"] = df["CC_MONTHS_COUNT"].notna().astype("int8")
    dicc.append({"feature": "TIENE_TARJETA", "grupo": "Presencia relacional", "formula": "1 si el cliente pertenece a la subpoblación con tarjeta de crédito (33.7%), 0 si no"})

    # 2. Ratios financieros sobre application_train
    inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    cred = df["AMT_CREDIT"].replace(0, np.nan)
    goods = df["AMT_GOODS_PRICE"].replace(0, np.nan)
    fam = df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    df["CREDIT_TO_INCOME_RATIO"] = (df["AMT_CREDIT"] / inc).astype("float32")
    dicc.append({"feature": "CREDIT_TO_INCOME_RATIO", "grupo": "Ratios financieros", "formula": "AMT_CREDIT / AMT_INCOME_TOTAL"})

    df["ANNUITY_TO_INCOME_RATIO"] = (df["AMT_ANNUITY"] / inc).astype("float32")
    dicc.append({"feature": "ANNUITY_TO_INCOME_RATIO", "grupo": "Ratios financieros", "formula": "AMT_ANNUITY / AMT_INCOME_TOTAL (carga financiera mensual sobre ingreso)"})

    df["ANNUITY_TO_CREDIT_RATIO"] = (df["AMT_ANNUITY"] / cred).astype("float32")
    dicc.append({"feature": "ANNUITY_TO_CREDIT_RATIO", "grupo": "Ratios financieros", "formula": "AMT_ANNUITY / AMT_CREDIT (inverso del plazo implícito del préstamo)"})

    df["CREDIT_TO_GOODS_RATIO"] = (df["AMT_CREDIT"] / goods).astype("float32")
    dicc.append({"feature": "CREDIT_TO_GOODS_RATIO", "grupo": "Ratios financieros", "formula": "AMT_CREDIT / AMT_GOODS_PRICE (sobre-financiamiento respecto al bien)"})

    df["INCOME_PER_FAM_MEMBER"] = (df["AMT_INCOME_TOTAL"] / fam).astype("float32")
    dicc.append({"feature": "INCOME_PER_FAM_MEMBER", "grupo": "Ratios financieros", "formula": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS"})

    # 3. Demográficas y estabilidad laboral
    df["EDAD_ANIOS"] = (-df["DAYS_BIRTH"] / 365.25).astype("float32")
    dicc.append({"feature": "EDAD_ANIOS", "grupo": "Temporales", "formula": "-DAYS_BIRTH / 365.25"})

    df["ANIOS_EMPLEO"] = (-df["DAYS_EMPLOYED"] / 365.25).astype("float32")
    dicc.append({"feature": "ANIOS_EMPLEO", "grupo": "Temporales", "formula": "-DAYS_EMPLOYED / 365.25 (NaN cuando FLAG_NO_EMPLEADO=1)"})

    df["EMPLEO_A_EDAD_RATIO"] = (df["ANIOS_EMPLEO"] / df["EDAD_ANIOS"].replace(0, np.nan)).astype("float32")
    dicc.append({"feature": "EMPLEO_A_EDAD_RATIO", "grupo": "Temporales", "formula": "ANIOS_EMPLEO / EDAD_ANIOS"})

    # 4. Combinaciones de EXT_SOURCE (las 3 variables con mayor señal individual)
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1).astype("float32")
    dicc.append({"feature": "EXT_SOURCE_MEAN", "grupo": "Scores externos", "formula": "Promedio de EXT_SOURCE_1, EXT_SOURCE_2 y EXT_SOURCE_3 disponibles"})

    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1).astype("float32")
    dicc.append({"feature": "EXT_SOURCE_MIN", "grupo": "Scores externos", "formula": "Mínimo entre EXT_SOURCE_1, EXT_SOURCE_2 y EXT_SOURCE_3"})

    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1).astype("float32")
    dicc.append({"feature": "EXT_SOURCE_MAX", "grupo": "Scores externos", "formula": "Máximo entre EXT_SOURCE_1, EXT_SOURCE_2 y EXT_SOURCE_3"})

    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1).astype("float32")
    dicc.append({"feature": "EXT_SOURCE_STD", "grupo": "Scores externos", "formula": "Desviación estándar entre los 3 scores externos"})

    df["EXT_SOURCE_PROD"] = (
        df["EXT_SOURCE_1"].fillna(0.5)
        * df["EXT_SOURCE_2"].fillna(0.5)
        * df["EXT_SOURCE_3"].fillna(0.5)
    ).astype("float32")
    dicc.append({"feature": "EXT_SOURCE_PROD", "grupo": "Scores externos", "formula": "Producto de EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3"})

    # 5. Síntesis de comportamiento de pago en tablas secundarias
    dpd_bureau = df["BUREAU_BB_DPD_MONTHS_SUM"].fillna(0)
    dpd_pos = df["POS_DPD_MONTHS_SUM"].fillna(0)
    dpd_inst = df["INST_LATE_COUNT"].fillna(0)
    dpd_cc = df["CC_DPD_MONTHS_SUM"].fillna(0)

    df["HIST_DPD_TOTAL_EVENTS"] = (dpd_bureau + dpd_pos + dpd_inst + dpd_cc).astype("float32")
    dicc.append({"feature": "HIST_DPD_TOTAL_EVENTS", "grupo": "Historial de pago", "formula": "Suma de eventos/meses con atraso en buró, POS, cuotas y tarjeta"})

    df["HIST_HAS_ANY_DPD"] = (df["HIST_DPD_TOTAL_EVENTS"] > 0).astype("int8")
    dicc.append({"feature": "HIST_HAS_ANY_DPD", "grupo": "Historial de pago", "formula": "1 si el cliente presenta al menos un atraso histórico en cualquier tabla, 0 si fue siempre puntual"})

    df["TOTAL_DEBT_TO_INCOME_RATIO"] = (
        (df["AMT_CREDIT"] + df["BUREAU_DEBT_SUM_TOTAL"].fillna(0)) / inc
    ).astype("float32")
    dicc.append({"feature": "TOTAL_DEBT_TO_INCOME_RATIO", "grupo": "Ratios financieros", "formula": "(AMT_CREDIT + deuda activa en buró) / AMT_INCOME_TOTAL"})

    return df, dicc


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa correlación con TARGET, porcentaje de nulos, varianza y rol (predictora vs
    auditoría de equidad) para producir `lista_features_final.csv` (`SCRUM-26`).
    """
    y = df["TARGET"].astype("float64")
    rows = []

    for col in df.columns:
        if col in ("SK_ID_CURR", "TARGET"):
            rows.append(
                {
                    "variable": col,
                    "tipo": str(df[col].dtype),
                    "pct_nulos": 0.0,
                    "corr_target_abs": np.nan,
                    "rol": "identificador" if col == "SK_ID_CURR" else "target",
                    "incluir_modelo": False,
                    "motivo": "Llave primaria / Variable objetivo",
                }
            )
            continue

        s = df[col]
        p_nulos = round(100.0 * float(s.isna().mean()), 4)
        vc = s.value_counts(dropna=False)
        p_dom = round(100.0 * float(vc.iloc[0]) / len(df), 4) if len(vc) else 100.0

        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            corr_val = float(s.astype("float64").corr(y))
            corr_abs = round(abs(corr_val), 5) if not np.isnan(corr_val) else 0.0
        else:
            codes = s.astype("category").cat.codes.replace(-1, np.nan)
            corr_val = float(codes.astype("float64").corr(y))
            corr_abs = round(abs(corr_val), 5) if not np.isnan(corr_val) else 0.0

        if col in PROTECTED_AUDIT_VARS:
            rol = "auditoria_sesgos"
            incluir = False
            motivo = "Variable protegida (Ley 29733 / SBS): se conserva para auditar paridad >= 0.8 en Fase 5, excluida como predictora directa"
        elif col.startswith("TOPEADO_"):
            rol = "predictora_bandera"
            incluir = True
            motivo = "Bandera de trazabilidad de topeo al percentil 99.9 (se conserva por diseño de reversibilidad Fase 3)"
        elif p_nulos >= 95.0:
            rol = "excluida_nulos"
            incluir = False
            motivo = f"Excluida por exceso de nulos ({p_nulos}% >= 95%)"
        elif p_dom >= 99.9:
            rol = "excluida_constante"
            incluir = False
            motivo = f"Excluida por ser casi constante (valor dominante {p_dom}% >= 99.9%)"
        else:
            rol = "predictora"
            incluir = True
            motivo = "Seleccionada para modelado (cumple umbrales de completitud, varianza y gobernanza)"

        rows.append(
            {
                "variable": col,
                "tipo": str(s.dtype),
                "pct_nulos": p_nulos,
                "corr_target_abs": corr_abs,
                "rol": rol,
                "incluir_modelo": incluir,
                "motivo": motivo,
            }
        )

    df_feat = pd.DataFrame(rows).sort_values(
        by=["incluir_modelo", "corr_target_abs"], ascending=[False, False]
    ).reset_index(drop=True)
    return df_feat


def run_build_dataset() -> pd.DataFrame:
    app_path = PROCESSED_DIR / "application_train_clean.parquet"
    bureau_path = PROCESSED_DIR / "bureau_agg.parquet"
    prev_pos_path = PROCESSED_DIR / "previous_pos_agg.parquet"
    cards_path = PROCESSED_DIR / "cc_installments_agg.parquet"

    print("[build_dataset] Cargando application_train_clean y las 3 tablas de agregados...")
    df = pd.read_parquet(app_path)
    bureau_agg = pd.read_parquet(bureau_path)
    prev_pos_agg = pd.read_parquet(prev_pos_path)
    cards_agg = pd.read_parquet(cards_path)

    print("[build_dataset] Consolidando por SK_ID_CURR (1 fila = 1 cliente)...")
    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    del bureau_agg
    gc.collect()

    df = df.merge(prev_pos_agg, on="SK_ID_CURR", how="left")
    del prev_pos_agg
    gc.collect()

    df = df.merge(cards_agg, on="SK_ID_CURR", how="left")
    del cards_agg
    gc.collect()

    assert len(df) == 307_511, f"Se esperaban 307,511 filas y se obtuvieron {len(df)}"
    assert df["SK_ID_CURR"].is_unique, "SK_ID_CURR debe seguir siendo único tras consolidar"

    print("[build_dataset] Creando variables derivadas (ratios, tendencias y scores)...")
    df, dicc_derivadas = add_derived_features(df)
    df = reduce_mem_usage(df, cat_exclude=["SK_ID_CURR", "TARGET"])

    # Guardar diccionario de features derivadas
    df_dicc = pd.DataFrame(dicc_derivadas)
    df_dicc.to_csv(FASE3_DIR / "diccionario_features_derivadas.csv", index=False)

    # Guardar solo el bloque de features derivadas + llave (requisito Cronograma v3)
    derived_cols = ["SK_ID_CURR"] + [d["feature"] for d in dicc_derivadas]
    df[derived_cols].to_parquet(PROCESSED_DIR / "features_derivadas.parquet", index=False)

    # Selección de variables
    print("[build_dataset] Evaluando selección de variables y exportando lista_features_final.csv...")
    df_feat = select_features(df)
    df_feat.to_csv(FASE3_DIR / "lista_features_final.csv", index=False)
    n_pred = int(df_feat["incluir_modelo"].sum())
    print(f"[build_dataset] Variables predictoras seleccionadas: {n_pred} de {len(df.columns) - 2}")

    out_consolidado = PROCESSED_DIR / "dataset_consolidado.parquet"
    df.to_parquet(out_consolidado, index=False)
    print(
        f"[build_dataset] Guardado {out_consolidado} "
        f"({df.shape[0]:,} filas x {df.shape[1]} columnas, {memory_mb(df)} MB en RAM)"
    )
    return df


if __name__ == "__main__":
    run_build_dataset()
