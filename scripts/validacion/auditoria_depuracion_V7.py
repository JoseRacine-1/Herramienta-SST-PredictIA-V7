"""Auditoría reproducible de las reglas de limpieza implementadas por V7.
Uso: python auditoria_depuracion_V7.py ruta/a/Estadisticas_Riesgos_Laborales_Positiva-_2025-2026.csv
No modifica el CSV ni entrena redes neuronales.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CAT_COLS = ["DPTO", "MPIO", "ACTIVEC"]
NUM_COLS_BASE = ["RELA_DEP", "RELA_INDEP", "AÑO_DE_INFORME", "MES_DE_INFORME"]
TARGETS = [
    "PRESUACCIDETRASUCE", "MUERTES_REPOR_AT", "NUEVAPENSIOINVA_R_AT",
    "NUEVAPENSIOINVA_R_EL", "INCAPERMAPARCIAR_AT", "INCAPERMAPARCIAR_EL"
]
NEEDED = CAT_COLS + NUM_COLS_BASE + TARGETS

def read_csv_robust(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin1")

def main():
    parser = argparse.ArgumentParser(description="Cuantifica las operaciones de depuración aplicadas en V7")
    parser.add_argument("csv", type=Path, help="CSV de entrada utilizado por la Herramienta V7")
    parser.add_argument("--salida", type=Path, default=Path("auditoria_depuracion_v7.xlsx"))
    args = parser.parse_args()
    raw = read_csv_robust(args.csv)
    raw.columns = raw.columns.str.strip()
    faltan = [c for c in NEEDED if c not in raw.columns]
    if faltan:
        raise ValueError(f"El CSV no contiene todas las columnas de V7: {faltan}")

    n_original = len(raw)
    pre = raw.copy()
    conversion = []
    for c in NUM_COLS_BASE + TARGETS:
        sin_analizar = pre[c].copy()
        convertido = pd.to_numeric(sin_analizar, errors="coerce")
        conversion.append({
            "variable": c,
            "faltantes_originales": int(sin_analizar.isna().sum()),
            "valores_no_nulos_no_convertibles": int((sin_analizar.notna() & convertido.isna()).sum()),
            "faltantes_despues_conversion": int(convertido.isna().sum()),
            "infinitos_despues_conversion": int(np.isinf(convertido).sum())
        })
        pre[c] = convertido

    pred_cols = CAT_COLS + NUM_COLS_BASE
    cat_missing = pre[CAT_COLS].isna().any(axis=1)
    num_missing = pre[NUM_COLS_BASE].isna().any(axis=1)
    incomplete = pre[pred_cols].isna().any(axis=1)
    kept = pre.loc[~incomplete].copy()
    target_stats = []
    for col in TARGETS:
        s = kept[col]
        target_stats.append({
            "variable": col,
            "faltantes_sustituidos_por_cero": int(s.isna().sum()),
            "negativos_convertidos_en_cero": int((s < 0).sum()),
            "ceros_originalmente_observados": int((s == 0).sum()),
            "ceros_despues_procesamiento": int((s.fillna(0).clip(lower=0) == 0).sum()),
            "valores_positivos_despues_procesamiento": int((s.fillna(0).clip(lower=0) > 0).sum()),
            "maximo_despues_procesamiento": float(s.fillna(0).clip(lower=0).max()) if len(kept) else None,
        })
        kept[col] = s.fillna(0).clip(lower=0)

    total_bruto = kept["RELA_DEP"] + kept["RELA_INDEP"]
    total_procesado = total_bruto.clip(lower=1)
    kept["TRABAJADORES"] = total_procesado
    kept["LOG_TRABAJADORES"] = np.log1p(total_procesado)
    base_stats = [{
        "variable": c,
        "faltantes_tras_eliminacion": int(kept[c].isna().sum()),
        "negativos_sin_modificacion": int((kept[c] < 0).sum()),
        "ceros_tras_eliminacion": int((kept[c] == 0).sum()),
    } for c in NUM_COLS_BASE]

    resumen = [
        ("registros_entrada_csv", n_original),
        ("registros_excluidos_por_predictores", int(incomplete.sum())),
        ("porcentaje_excluido", float(incomplete.mean() * 100) if n_original else 0),
        ("registros_excluidos_con_faltantes_categoricos", int(cat_missing.sum())),
        ("registros_excluidos_con_faltantes_numericos", int(num_missing.sum())),
        ("registros_excluidos_con_ambos_tipos", int((cat_missing & num_missing).sum())),
        ("registros_conservados", len(kept)),
        ("porcentaje_conservado", float(len(kept) / n_original * 100) if n_original else 0),
        ("valores_objetivo_faltantes_sustituidos_total", sum(x["faltantes_sustituidos_por_cero"] for x in target_stats)),
        ("valores_objetivo_negativos_corregidos_total", sum(x["negativos_convertidos_en_cero"] for x in target_stats)),
        ("registros_con_total_trabajadores_inferior_a_uno", int((total_bruto < 1).sum())),
        ("registros_duplicados_exactos_entrada_solo_diagnostico", int(raw.duplicated().sum())),
        ("registros_con_mes_fuera_de_1_a_12_conservados", int((~kept["MES_DE_INFORME"].between(1, 12)).sum())),
    ]
    reporte_resumen = pd.DataFrame(resumen, columns=["indicador", "resultado"])
    for item in target_stats:
        print(f'{item["variable"]}: faltantes a cero={item["faltantes_sustituidos_por_cero"]}; negativos a cero={item["negativos_convertidos_en_cero"]}')
    print(reporte_resumen.to_string(index=False))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.salida) as writer:
        reporte_resumen.to_excel(writer, sheet_name="resumen_depuracion", index=False)
        pd.DataFrame(conversion).to_excel(writer, sheet_name="conversion_numerica", index=False)
        pd.DataFrame(target_stats).to_excel(writer, sheet_name="tratamiento_objetivos", index=False)
        pd.DataFrame(base_stats).to_excel(writer, sheet_name="predictores_numericos", index=False)
        pd.DataFrame([{"variable": c, "faltantes_entrada": int(pre[c].isna().sum())} for c in pred_cols]).to_excel(writer, sheet_name="predictores_faltantes", index=False)
    print("Auditoría guardada en:", args.salida.resolve())

if __name__ == "__main__":
    main()
