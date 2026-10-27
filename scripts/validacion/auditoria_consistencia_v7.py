"""Segunda auditoría independiente de consistencia para la Herramienta V7.

Analiza la base CSV postunificación sin modificarla ni reentrenar modelos.
Replica el filtro y las reglas de limpieza de la función load_and_clean() de V7.

Uso en Windows / terminal:
    python auditoria_consistencia_v7.py "C:\\ruta\\a\\Estadisticas_Riesgos_Laborales_Positiva-_2025-2026.csv"

Resultado: se crea una carpeta auditoria_consistencia_v7 junto al CSV, con
resúmenes CSV compatibles con Excel y un informe TXT.
Requiere pandas y numpy (ya utilizadas por Herramienta V7).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CAT_COLS = ["DPTO", "MPIO", "ACTIVEC"]
NUM_COLS_BASE = ["RELA_DEP", "RELA_INDEP", "AÑO_DE_INFORME", "MES_DE_INFORME"]
TARGETS = [
    "PRESUACCIDETRASUCE",
    "MUERTES_REPOR_AT",
    "NUEVAPENSIOINVA_R_AT",
    "NUEVAPENSIOINVA_R_EL",
    "INCAPERMAPARCIAR_AT",
    "INCAPERMAPARCIAR_EL",
]
DETAIL = ["fila_csv"] + CAT_COLS + ["AÑO_DE_INFORME", "MES_DE_INFORME",
    "RELA_DEP", "RELA_INDEP", "TRABAJADORES_SUMA_ORIGINAL",
    "TRABAJADORES_AJUSTADOS"] + TARGETS


def load_csv(path: Path) -> pd.DataFrame:
    try:
        out = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        out = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
    out.columns = out.columns.str.strip()
    return out


def export_csv(df: pd.DataFrame, dest: Path):
    df.to_csv(dest, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Cruza trabajadores registrados y eventos laborales en la base V7")
    parser.add_argument("csv", type=Path, help="Ruta del CSV postunificación que utiliza Herramienta V7")
    parser.add_argument("--salida", type=Path, default=None, help="Carpeta de salida opcional")
    args = parser.parse_args()
    source = args.csv.expanduser().resolve()
    if not source.is_file():
        parser.error(f"No se encuentra el archivo CSV: {source}")
    out_dir = (args.salida or (source.parent / "auditoria_consistencia_v7")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_csv(source)
    missing = [c for c in CAT_COLS + NUM_COLS_BASE + TARGETS if c not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas que requiere V7: {missing}")
    n_input = len(raw)
    df = raw.copy()
    df["fila_csv"] = np.arange(2, n_input + 2)
    for c in NUM_COLS_BASE + TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=CAT_COLS + NUM_COLS_BASE).copy()
    for c in TARGETS:
        df[c] = df[c].fillna(0).clip(lower=0)

    df["TRABAJADORES_SUMA_ORIGINAL"] = df["RELA_DEP"] + df["RELA_INDEP"]
    df["TRABAJADORES_AJUSTADOS"] = df["TRABAJADORES_SUMA_ORIGINAL"].clip(lower=1)
    low = df["TRABAJADORES_SUMA_ORIGINAL"] < 1
    exact_zero = df["TRABAJADORES_SUMA_ORIGINAL"] == 0
    negative = df["TRABAJADORES_SUMA_ORIGINAL"] < 0
    event_any = df[TARGETS].gt(0).any(axis=1)
    accident_pos = df["PRESUACCIDETRASUCE"] > 0
    death_pos = df["MUERTES_REPOR_AT"] > 0
    n = len(df)

    metrics = [
        ("Archivo CSV analizado", source.name),
        ("Registros iniciales", n_input),
        ("Registros después de excluir predictores incompletos", n),
        ("Registros con suma de trabajadores <1", int(low.sum())),
        ("Registros con suma de trabajadores exactamente 0", int(exact_zero.sum())),
        ("Registros con suma de trabajadores negativa", int(negative.sum())),
        ("Porcentaje con trabajadores <1 sobre registros procesados", round(float(low.mean() * 100), 4) if n else 0),
        ("Trabajadores <1 y al menos un evento positivo", int((low & event_any).sum())),
        ("Trabajadores =0 y al menos un evento positivo", int((exact_zero & event_any).sum())),
        ("Trabajadores <1 y accidente positivo", int((low & accident_pos).sum())),
        ("Trabajadores <1 y muerte positiva", int((low & death_pos).sum())),
        ("Todos los registros con muertes positivas", int(death_pos.sum())),
        ("Mayor número reportado de muertes en un registro", float(df["MUERTES_REPOR_AT"].max()) if n else None),
        ("Registros con muertes > accidentes (alerta para revisión, no error demostrado)", int((df["MUERTES_REPOR_AT"] > df["PRESUACCIDETRASUCE"]).sum())),
        ("Registros con trabajadores originales dependientes negativos", int((df["RELA_DEP"] < 0).sum())),
        ("Registros con trabajadores originales independientes negativos", int((df["RELA_INDEP"] < 0).sum())),
    ]
    summary = pd.DataFrame(metrics, columns=["indicador", "resultado"])
    export_csv(summary, out_dir / "01_resumen_consistencia.csv")

    cross = []
    for tgt in TARGETS:
        event_pos = df[tgt] > 0
        cross.append({
            "variable_objetivo": tgt,
            "registros_evento_positivo_total": int(event_pos.sum()),
            "registros_evento_positivo_y_trabajadores_inferior_1": int((event_pos & low).sum()),
            "registros_evento_positivo_y_trabajadores_igual_0": int((event_pos & exact_zero).sum()),
            "cantidad_eventos_en_registros_trabajadores_inferior_1": float(df.loc[low, tgt].sum()),
            "cantidad_eventos_en_registros_trabajadores_igual_0": float(df.loc[exact_zero, tgt].sum()),
            "max_eventos_en_registros_trabajadores_inferior_1": float(df.loc[low, tgt].max()) if low.any() else 0,
            "max_eventos_en_todos_los_registros": float(df[tgt].max()) if n else 0,
        })
    export_csv(pd.DataFrame(cross), out_dir / "02_cruces_por_desenlace.csv")

    critical = df.loc[low & event_any, DETAIL].copy()
    critical = critical.sort_values(["MUERTES_REPOR_AT", "PRESUACCIDETRASUCE"], ascending=[False, False])
    export_csv(critical, out_dir / "03_trabajadores_bajos_con_eventos.csv")

    graves = df.loc[death_pos, DETAIL].copy()
    graves = graves.sort_values(["MUERTES_REPOR_AT", "PRESUACCIDETRASUCE"], ascending=[False, False])
    export_csv(graves.head(500), out_dir / "04_500_registros_mas_altos_muertes.csv")

    max_death = df["MUERTES_REPOR_AT"].max() if n else np.nan
    export_csv(df.loc[df["MUERTES_REPOR_AT"] == max_death, DETAIL], out_dir / "05_registros_maximo_muertes.csv")

    period = df.groupby(["AÑO_DE_INFORME", "MES_DE_INFORME"], dropna=False).agg(
        registros=("fila_csv", "size"),
        registros_trabajadores_inferior_1=("TRABAJADORES_SUMA_ORIGINAL", lambda s: int((s < 1).sum())),
        registros_eventos_positivos=("PRESUACCIDETRASUCE", lambda s: int((s > 0).sum())),
        accidentes_total=("PRESUACCIDETRASUCE", "sum"),
        muertes_total=("MUERTES_REPOR_AT", "sum"),
        max_muertes_un_registro=("MUERTES_REPOR_AT", "max"),
    ).reset_index()
    export_csv(period, out_dir / "06_resumen_por_periodo.csv")

    report = [
        "AUDITORÍA DE CONSISTENCIA V7 — SEGUNDA COMPROBACIÓN",
        f"Archivo: {source.name}",
        f"Registros leídos: {n_input:,}; conservados según load_and_clean de V7: {n:,}",
        f"Registros con trabajadores <1: {int(low.sum()):,} ({low.mean()*100:.2f} %)",
        f"De ellos, con algún evento positivo: {int((low & event_any).sum()):,}",
        f"Con accidentes positivos: {int((low & accident_pos).sum()):,}; con muertes positivas: {int((low & death_pos).sum()):,}",
        f"Máximo de muertes por registro: {max_death}",
        "",
        "INTERPRETACIÓN: Coincidir cero trabajadores y eventos positivos es una ALERTA para verificar la fuente,",
        "no demuestra por sí solo que exista un error (puede haber diferencias de temporalidad o definición).",
        "No corregir ni eliminar observaciones hasta verificar su significado y origen.",
        "Los valores de trabajadores corresponden al registro mensual, no a personas únicas del periodo.",
        "",
        "Archivos: 01 resumen; 02 cruces por desenlace; 03 todos los casos con trabajadores <1 y eventos;",
        "04 los 500 registros con muertes más altas; 05 máximo de muertes; 06 resumen por año-mes.",
        "La columna fila_csv indica el número de fila aproximado de origen (primera fila=encabezado).",
        "El procedimiento replica la limpieza programada en V7, no la unificación previa del CSV.",
    ]
    (out_dir / "00_LEEME_informe.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nCarpeta de resultados:", out_dir)


if __name__ == "__main__":
    main()
