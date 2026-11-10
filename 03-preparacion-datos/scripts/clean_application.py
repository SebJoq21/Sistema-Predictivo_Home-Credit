"""
3.3 y 3.4 Limpieza de `application_train` y tratamiento de valores faltantes como
información implícita (`SCRUM-18` y `SCRUM-19`).
Reproduce exactamente la lógica validada en `fase3_preparacion.ipynb` y exporta:
- `data/processed/application_train_clean.parquet`
-Todas las tablas CSV de auditoría en `03-preparacion-datos/salida_fase3/tablas/`
-Las 7 figuras de alta resolución en `03-preparacion-datos/figuras/`
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.memory_optimization import memory_mb, reduce_mem_usage

RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
FASE3_DIR = Path("03-preparacion-datos")
OUT_DIR = FASE3_DIR / "salida_fase3"
TABLAS_DIR = OUT_DIR / "tablas"
FIGURAS_DIR = FASE3_DIR / "figuras"

CHUNKSIZE = 1_000_000
UMBRAL_VACIO = 95.0
UMBRAL_CONSTANTE = 99.9
UMBRAL_BANDERA = 5.0

OUTLIERS_APP = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "OWN_CAR_AGE",
]
TOPEAR_P999 = [
    "AMT_INCOME_TOTAL",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_QRT",
]
SENTINELA_EMPLEO = 365243
XNA_A_NAN = ["CODE_GENDER", "ORGANIZATION_TYPE"]


def pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 4) if b else float("nan")


def resumen_num(x: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(x, dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {}
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    p1, p50, p99, p999 = np.percentile(arr, [1, 50, 99, 99.9])
    b15 = int(((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)).sum())
    b30 = int(((arr < q1 - 3.0 * iqr) | (arr > q3 + 3.0 * iqr)).sum())
    return dict(
        n=int(arr.size),
        min=float(arr.min()),
        p1=round(float(p1), 2),
        mediana=round(float(p50), 2),
        p99=round(float(p99), 2),
        p99_9=round(float(p999), 2),
        max=float(arr.max()),
        q1=round(float(q1), 2),
        q3=round(float(q3), 2),
        n_fuera_1_5IQR=b15,
        pct_fuera_1_5IQR=pct(b15, arr.size),
        n_fuera_3IQR=b30,
        pct_fuera_3IQR=pct(b30, arr.size),
    )


def run_clean_application() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    print("[clean_application] Cargando application_train.csv...")
    df = pd.read_csv(RAW_DIR / "application_train.csv", low_memory=False)
    n = len(df)

    # 1. Análisis de nulos y bloques con patrón idéntico
    nulos = df.isna().sum()
    tab_nulos = (
        pd.DataFrame(
            {
                "columna": nulos.index,
                "nulos": nulos.values,
                "pct_nulos": [pct(int(v), n) for v in nulos.values],
            }
        )
        .sort_values("nulos", ascending=False)
        .reset_index(drop=True)
    )
    tab_nulos.to_csv(TABLAS_DIR / "nulos_application_train.csv", index=False)

    con_nulos = tab_nulos[tab_nulos["nulos"] > 0]["columna"].tolist()
    firmas: dict[bytes, list[str]] = {}
    for c in con_nulos:
        h = pd.util.hash_pandas_object(df[c].isna(), index=False).to_numpy().tobytes()
        firmas.setdefault(h, []).append(c)

    bloques = [
        {
            "bloque": i,
            "n_columnas": len(cs),
            "pct_nulos": pct(int(df[cs[0]].isna().sum()), n),
            "columnas": ", ".join(cs),
        }
        for i, cs in enumerate(
            sorted(firmas.values(), key=lambda x: -len(x)), 1
        )
    ]
    pd.DataFrame(bloques).to_csv(
        TABLAS_DIR / "faltantes_bloques_patron_identico.csv", index=False
    )

    # 2. Ausencia vs TARGET (chi2)
    vs = []
    for c in tab_nulos[tab_nulos["pct_nulos"] >= 1.0]["columna"]:
        m = df[c].isna()
        t1, t0 = df.loc[m, "TARGET"].mean() * 100, df.loc[~m, "TARGET"].mean() * 100
        ct = pd.crosstab(m, df["TARGET"])
        p_val = float(stats.chi2_contingency(ct)[1]) if ct.shape == (2, 2) else float("nan")
        vs.append(
            {
                "columna": c,
                "pct_nulos": pct(int(m.sum()), n),
                "mora_cuando_falta_%": round(float(t1), 3),
                "mora_cuando_existe_%": round(float(t0), 3),
                "diferencia_pp": round(float(t1 - t0), 3),
                "p_chi2": p_val,
            }
        )
    df_vs = pd.DataFrame(vs).sort_values("diferencia_pp", key=lambda s: s.abs(), ascending=False)
    df_vs.to_csv(TABLAS_DIR / "faltantes_vs_target.csv", index=False)

    n_fila = df.isna().sum(axis=1)
    tramos = pd.cut(n_fila, bins=[-1, 0, 5, 10, 20, 40, 200])
    por_fila = (
        df.groupby(tramos, observed=False)["TARGET"]
        .agg(filas="count", tasa_mora=lambda s: round(float(s.mean() * 100), 3))
        .reset_index()
        .rename(columns={"index": "nulos_por_fila"})
    )
    por_fila.columns = ["nulos_por_fila", "filas", "tasa_mora"]
    por_fila.to_csv(TABLAS_DIR / "faltantes_por_fila.csv", index=False)

    # Figuras 1 a 4
    top20 = tab_nulos.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top20["columna"], top20["pct_nulos"], color="#2b5c8f")
    ax.set_xlabel("% de valores faltantes")
    ax.set_title("Top 20 columnas con más nulos — application_train")
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_nulos_top20.png", dpi=140)
    plt.close(fig)

    top_c = tab_nulos[tab_nulos["nulos"] > 0].head(20)["columna"].tolist()
    corr = df[top_c].isna().astype("int8").corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_c)))
    ax.set_xticklabels(top_c, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_c)))
    ax.set_yticklabels(top_c, fontsize=7)
    ax.set_title("Correlación entre indicadores de ausencia (top 20 nulos)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_patron_faltantes_corr.png", dpi=140)
    plt.close(fig)

    top_vs = df_vs.head(15).iloc[::-1]
    y = np.arange(len(top_vs))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(y - 0.18, top_vs["mora_cuando_existe_%"], height=0.35, label="Dato presente", color="#2b5c8f")
    ax.barh(y + 0.18, top_vs["mora_cuando_falta_%"], height=0.35, label="Dato ausente", color="#d95f02")
    ax.set_yticks(y)
    ax.set_yticklabels(top_vs["columna"], fontsize=8)
    ax.set_xlabel("Tasa de mora (%)")
    ax.set_title("Tasa de mora según presencia o ausencia del dato")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_faltante_vs_target.png", dpi=140)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    etq = por_fila["nulos_por_fila"].astype(str)
    ax1.bar(etq, por_fila["filas"], color="#9ecae1")
    ax1.set_ylabel("Solicitudes")
    ax1.set_xlabel("Rango de columnas nulas por fila")
    ax2 = ax1.twinx()
    ax2.plot(etq, por_fila["tasa_mora"], color="#d95f02", marker="o", linewidth=2)
    ax2.set_ylabel("Tasa de mora (%)")
    ax1.set_title("Cantidad de nulos por solicitud vs tasa de mora")
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_nulos_por_fila.png", dpi=140)
    plt.close(fig)

    # 3. Análisis de atípicos
    filas_atip = []
    for c in OUTLIERS_APP:
        s = df[c].replace(SENTINELA_EMPLEO, np.nan) if c == "DAYS_EMPLOYED" else df[c]
        r = resumen_num(s)
        if r:
            filas_atip.append({"columna": c, **r})
    df_at = pd.DataFrame(filas_atip)
    df_at.to_csv(TABLAS_DIR / "atipicos_application_train.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    d_ord = df_at.sort_values("pct_fuera_3IQR")
    ax.barh(d_ord["columna"], d_ord["pct_fuera_3IQR"], color="#c0392b", label="> 3x IQR (extremo)")
    ax.barh(d_ord["columna"], d_ord["pct_fuera_1_5IQR"], color="#f39c12", alpha=0.45, label="> 1.5x IQR")
    ax.set_xlabel("% de filas")
    ax.set_title("Valores atípicos por variable (regla IQR de Tukey)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_atipicos_pct.png", dpi=140)
    plt.close(fig)

    emp = -df.loc[df["DAYS_EMPLOYED"] != SENTINELA_EMPLEO, "DAYS_EMPLOYED"] / 365.25
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(emp.dropna(), bins=50, color="#2b5c8f", edgecolor="white")
    ax.set_xlabel("Años de empleo (excluido centinela 365243)")
    ax.set_ylabel("Solicitudes")
    ax.set_title("Distribución de antigüedad laboral tras retirar el centinela")
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_days_employed.png", dpi=140)
    plt.close(fig)

    # 4. Duplicados e inconsistencias
    pd.DataFrame(
        [
            {
                "filas": n,
                "duplicados_fila_completa": int(df.duplicated().sum()),
                "SK_ID_CURR_repetidos": int(df["SK_ID_CURR"].duplicated().sum()),
            }
        ]
    ).to_csv(TABLAS_DIR / "duplicados_application_train.csv", index=False)

    de = df["DAYS_EMPLOYED"]
    db = df["DAYS_BIRTH"]
    reglas = [
        ("DAYS_BIRTH >= 0 (debe ser negativo)", db >= 0),
        ("Edad < 18 o > 100 años", ((-db / 365.25) < 18) | ((-db / 365.25) > 100)),
        ("DAYS_EMPLOYED = centinela 365243", de == SENTINELA_EMPLEO),
        ("DAYS_EMPLOYED positivo distinto del centinela", (de > 0) & (de != SENTINELA_EMPLEO)),
        ("Años empleado > edad", (de < 0) & ((-de) > (-db))),
        ("Empezó a trabajar antes de los 14 años", (de < 0) & (((-db) - (-de)) < 14 * 365.25)),
        ("CODE_GENDER = XNA", df["CODE_GENDER"] == "XNA"),
        ("ORGANIZATION_TYPE = XNA", df["ORGANIZATION_TYPE"] == "XNA"),
        (
            "ORGANIZATION_TYPE = XNA y centinela de empleo coinciden",
            (df["ORGANIZATION_TYPE"] == "XNA") & (de == SENTINELA_EMPLEO),
        ),
        ("CNT_FAM_MEMBERS < CNT_CHILDREN", df["CNT_FAM_MEMBERS"] < df["CNT_CHILDREN"]),
        ("AMT_INCOME_TOTAL <= 0", df["AMT_INCOME_TOTAL"] <= 0),
        ("AMT_CREDIT <= 0", df["AMT_CREDIT"] <= 0),
        ("AMT_ANNUITY > AMT_CREDIT", df["AMT_ANNUITY"] > df["AMT_CREDIT"]),
        ("FLAG_OWN_CAR = N pero OWN_CAR_AGE informado", (df["FLAG_OWN_CAR"] == "N") & df["OWN_CAR_AGE"].notna()),
        ("FLAG_OWN_CAR = Y pero OWN_CAR_AGE vacío", (df["FLAG_OWN_CAR"] == "Y") & df["OWN_CAR_AGE"].isna()),
        ("EXT_SOURCE_1 fuera de [0,1]", (df["EXT_SOURCE_1"] < 0) | (df["EXT_SOURCE_1"] > 1)),
        ("EXT_SOURCE_2 fuera de [0,1]", (df["EXT_SOURCE_2"] < 0) | (df["EXT_SOURCE_2"] > 1)),
        ("EXT_SOURCE_3 fuera de [0,1]", (df["EXT_SOURCE_3"] < 0) | (df["EXT_SOURCE_3"] > 1)),
        ("DAYS_REGISTRATION > 0", df["DAYS_REGISTRATION"] > 0),
        ("DAYS_ID_PUBLISH > 0", df["DAYS_ID_PUBLISH"] > 0),
    ]
    inc = [
        {"chequeo": nom, "filas_afectadas": int(m.sum()), "pct": pct(int(m.sum()), n)}
        for nom, m in reglas
    ]
    pd.DataFrame(inc).to_csv(TABLAS_DIR / "inconsistencias_application_train.csv", index=False)

    # 5. Selección preliminar de columnas casi constantes / casi vacías
    filas_excl = []
    for c in df.columns:
        if c in ("SK_ID_CURR", "TARGET"):
            continue
        p_nul = pct(int(df[c].isna().sum()), n)
        vc = df[c].value_counts(dropna=False)
        p_dom = pct(int(vc.iloc[0]), n) if len(vc) else 0.0
        if p_nul >= UMBRAL_VACIO:
            filas_excl.append({"tabla": "application_train", "columna": c, "motivo": f">= {UMBRAL_VACIO}% nulos ({p_nul}%)", "decision": "excluir"})
        elif p_dom >= UMBRAL_CONSTANTE:
            filas_excl.append({"tabla": "application_train", "columna": c, "motivo": f"casi constante (valor dominante {p_dom}%)", "decision": "excluir"})
    filas_excl.append({"tabla": "previous_application", "columna": "RATE_INTEREST_PRIMARY", "motivo": "99.64% de nulos", "decision": "excluir"})
    filas_excl.append({"tabla": "previous_application", "columna": "RATE_INTEREST_PRIVILEGED", "motivo": "99.64% de nulos", "decision": "excluir"})
    df_excl = pd.DataFrame(filas_excl)
    df_excl.to_csv(TABLAS_DIR / "seleccion_columnas_excluidas.csv", index=False)

    # 6. Aplicación de limpieza v1 y creación de banderas FALTA_*
    mem_antes = memory_mb(df)
    shape_antes = df.shape
    nulos_antes = int(df.isna().sum().sum())
    df_clean = df.copy()
    bitacora = []

    a_excluir = [
        c for c in df_excl[df_excl["tabla"] == "application_train"]["columna"]
        if c in df_clean.columns
    ]
    if a_excluir:
        df_clean.drop(columns=a_excluir, inplace=True)
        bitacora.append({"paso": "columnas excluidas", "detalle": ", ".join(a_excluir), "filas": len(a_excluir)})

    m_cent = df_clean["DAYS_EMPLOYED"] == SENTINELA_EMPLEO
    df_clean["FLAG_NO_EMPLEADO"] = m_cent.astype("int8")
    df_clean.loc[m_cent, "DAYS_EMPLOYED"] = np.nan
    bitacora.append({"paso": "DAYS_EMPLOYED centinela -> NaN + FLAG_NO_EMPLEADO", "detalle": "365243 -> NaN", "filas": int(m_cent.sum())})

    for c in XNA_A_NAN:
        if c in df_clean.columns:
            m = df_clean[c] == "XNA"
            df_clean.loc[m, c] = np.nan
            bitacora.append({"paso": f"{c}: XNA -> NaN", "detalle": "XNA -> NaN", "filas": int(m.sum())})

    topeos = []
    for c in TOPEAR_P999:
        if c not in df_clean.columns:
            continue
        s = df_clean[c]
        tope = float(s.quantile(0.999))
        m = s > tope
        antes_m, antes_sd, antes_max = float(s.mean()), float(s.std()), float(s.max())
        df_clean[f"TOPEADO_{c}"] = m.astype("int8")
        df_clean.loc[m, c] = tope
        topeos.append(
            {
                "columna": c,
                "tope_p99_9": round(tope, 2),
                "filas_topeadas": int(m.sum()),
                "max_antes": antes_max,
                "max_despues": float(df_clean[c].max()),
                "media_antes": round(antes_m, 2),
                "media_despues": round(float(df_clean[c].mean()), 2),
                "desv_antes": round(antes_sd, 2),
                "desv_despues": round(float(df_clean[c].std()), 2),
            }
        )
        bitacora.append({"paso": f"{c}: topeo p99.9", "detalle": f"tope={tope:.2f}", "filas": int(m.sum())})
    pd.DataFrame(topeos).to_csv(TABLAS_DIR / "topeo_antes_despues.csv", index=False)

    # Figura 7: ingreso antes y después
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(df["AMT_INCOME_TOTAL"].dropna() / 1e6, bins=50, color="#c0392b")
    axes[0].set_title("Ingreso antes (millones)")
    axes[0].set_yscale("log")
    axes[1].hist(df_clean["AMT_INCOME_TOTAL"].dropna() / 1e3, bins=50, color="#2b5c8f")
    axes[1].set_title("Ingreso después del topeo p99.9 (miles)")
    plt.tight_layout()
    fig.savefig(FIGURAS_DIR / "F_ingreso_antes_despues.png", dpi=140)
    plt.close(fig)

    # Banderas de ausencia por bloque (sin duplicar por bloque idéntico)
    banderas = []
    nuevas_cols = {}
    for cs in firmas.values():
        cs_vivas = [c for c in cs if c in df_clean.columns]
        if not cs_vivas:
            continue
        c0 = cs_vivas[0]
        p = pct(int(df_clean[c0].isna().sum()), len(df_clean))
        if p >= UMBRAL_BANDERA:
            nom = f"FALTA_{c0}"
            nuevas_cols[nom] = df_clean[c0].isna().astype("int8")
            banderas.append({"bandera": nom, "cubre_columnas": len(cs_vivas), "pct_ausente": p})
    if nuevas_cols:
        df_clean = pd.concat([df_clean, pd.DataFrame(nuevas_cols, index=df_clean.index)], axis=1)
    pd.DataFrame(banderas).to_csv(TABLAS_DIR / "banderas_ausencia_creadas.csv", index=False)
    bitacora.append({"paso": "banderas FALTA_*", "detalle": f"{len(banderas)} banderas (umbral {UMBRAL_BANDERA}%)", "filas": len(banderas)})

    # Downcast de tipos
    df_clean = reduce_mem_usage(df_clean, cat_exclude=["SK_ID_CURR", "TARGET"])
    mem_despues = memory_mb(df_clean)

    pd.DataFrame(bitacora).to_csv(TABLAS_DIR / "registro_limpieza.csv", index=False)
    comp = pd.DataFrame(
        [
            {"momento": "antes", "filas": shape_antes[0], "columnas": shape_antes[1], "nulos_totales": nulos_antes, "memoria_MB": mem_antes},
            {"momento": "después", "filas": df_clean.shape[0], "columnas": df_clean.shape[1], "nulos_totales": int(df_clean.isna().sum().sum()), "memoria_MB": mem_despues},
        ]
    )
    comp.to_csv(TABLAS_DIR / "antes_despues_application_train.csv", index=False)

    out_parquet = PROCESSED_DIR / "application_train_clean.parquet"
    df_clean.to_parquet(out_parquet, index=False)
    print(f"[clean_application] Guardado {out_parquet} ({df_clean.shape}, {mem_despues} MB)")

    del df
    gc.collect()
    return df_clean


if __name__ == "__main__":
    run_clean_application()
