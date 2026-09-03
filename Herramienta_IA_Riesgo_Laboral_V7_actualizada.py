import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance,
    r2_score,
    explained_variance_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN PORTABLE
# ============================================================

def detectar_raiz_proyecto():
    """
    Detecta automáticamente la carpeta raíz del proyecto.
    La carpeta raíz esperada es CODE.
    """

    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd().resolve()

    posibles_bases = [base] + list(base.parents)

    # Caso 1: el notebook/script se ejecuta dentro de CODE
    for carpeta in posibles_bases:
        if (
            (carpeta / "ARCHIVOS PROYECTO").exists()
            or (carpeta / "archivos_proyecto").exists()
        ):
            return carpeta

    # Caso 2: Jupyter se abrió desde la carpeta Python y CODE está dentro
    for carpeta in posibles_bases:
        code_dir = carpeta / "CODE"
        if (
            (code_dir / "ARCHIVOS PROYECTO").exists()
            or (code_dir / "archivos_proyecto").exists()
        ):
            return code_dir

    raise FileNotFoundError(
        "No se pudo detectar la carpeta raíz del proyecto. "
        "Verifique que exista la carpeta CODE y que dentro esté "
        "'ARCHIVOS PROYECTO' o 'archivos_proyecto'."
    )


def detectar_carpeta_archivos_proyecto(project_root):
    """
    Detecta la carpeta donde están los archivos del proyecto.
    Acepta tanto el nombre actual como una versión más profesional sin espacios.
    """

    candidatos = [
        project_root / "ARCHIVOS PROYECTO",
        project_root / "archivos_proyecto"
    ]

    for carpeta in candidatos:
        if carpeta.exists():
            return carpeta

    raise FileNotFoundError(
        "No se encontró la carpeta de archivos del proyecto. "
        "Debe existir 'ARCHIVOS PROYECTO' o 'archivos_proyecto' dentro de CODE."
    )


def detectar_carpeta_base_limpia(archivos_proyecto_dir):
    """
    Detecta la carpeta donde está la base limpia postunificación.
    Acepta diferentes nombres para evitar fallos por espacios, mayúsculas o renombramientos.
    """

    candidatos = [
        archivos_proyecto_dir / "Base de datos limpia postunificacion",
        archivos_proyecto_dir / "base_limpia_postunificacion",
        archivos_proyecto_dir / "Base_de_datos_limpia_postunificacion",
        archivos_proyecto_dir / "base de datos limpia postunificacion"
    ]

    for carpeta in candidatos:
        if carpeta.exists():
            return carpeta

    raise FileNotFoundError(
        "No se encontró la carpeta de base limpia postunificación. "
        "Revise que dentro de ARCHIVOS PROYECTO exista la carpeta "
        "'Base de datos limpia postunificacion'."
    )


def detectar_archivo_csv(data_dir):
    """
    Detecta automáticamente el archivo CSV de entrada.
    Primero busca nombres esperados y luego cualquier CSV relacionado con Positiva 2025-2026.
    """

    nombres_esperados = [
        "Estadisticas_Riesgos_Laborales_Positiva-_2025-2026.csv",
        "Estadisticas_Riegos_Laborales_Positiva-_2025-2026.csv"
    ]

    for nombre in nombres_esperados:
        archivo = data_dir / nombre
        if archivo.exists():
            return archivo

    csv_files = list(data_dir.glob("*.csv"))

    candidatos = [
        archivo for archivo in csv_files
        if "positiva" in archivo.name.lower()
        and "2025" in archivo.name.lower()
        and "2026" in archivo.name.lower()
    ]

    if len(candidatos) == 1:
        return candidatos[0]

    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "No se pudo identificar de forma única el archivo CSV de entrada.\n"
        f"Carpeta revisada: {data_dir}\n"
        f"Archivos CSV encontrados: {[archivo.name for archivo in csv_files]}\n"
        "Deje un solo CSV en la carpeta de base limpia o use uno de los nombres esperados."
    )


PROJECT_ROOT = detectar_raiz_proyecto()
ARCHIVOS_PROYECTO_DIR = detectar_carpeta_archivos_proyecto(PROJECT_ROOT)
DATA_DIR = detectar_carpeta_base_limpia(ARCHIVOS_PROYECTO_DIR)
FILE_PATH = detectar_archivo_csv(DATA_DIR)

OUTPUT_DIR = PROJECT_ROOT / "resultados_herramienta_sst_v7_dashboard"

EXCEL_DIR = OUTPUT_DIR / "excel"
GRAFICAS_DIR = OUTPUT_DIR / "graficas"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
LOGS_DIR = OUTPUT_DIR / "logs"

for directory in [OUTPUT_DIR, EXCEL_DIR, GRAFICAS_DIR, DASHBOARD_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "resumen_ejecucion_v7.txt"

print("Raíz del proyecto detectada:", PROJECT_ROOT)
print("Carpeta de archivos del proyecto:", ARCHIVOS_PROYECTO_DIR)
print("Carpeta de base limpia:", DATA_DIR)
print("Base de datos utilizada:", FILE_PATH)
print("Carpeta de resultados:", OUTPUT_DIR)


CAT_COLS = ["DPTO", "MPIO", "ACTIVEC"]
NUM_COLS_BASE = ["RELA_DEP", "RELA_INDEP", "AÑO_DE_INFORME", "MES_DE_INFORME"]

TARGETS = [
    "PRESUACCIDETRASUCE",
    "MUERTES_REPOR_AT",
    "NUEVAPENSIOINVA_R_AT",
    "NUEVAPENSIOINVA_R_EL",
    "INCAPERMAPARCIAR_AT",
    "INCAPERMAPARCIAR_EL"
]

PESOS_SEVERIDAD = {
    "PRESUACCIDETRASUCE": 1.0,
    "INCAPERMAPARCIAR_AT": 3.0,
    "INCAPERMAPARCIAR_EL": 3.0,
    "NUEVAPENSIOINVA_R_AT": 4.0,
    "NUEVAPENSIOINVA_R_EL": 4.0,
    "MUERTES_REPOR_AT": 5.0
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)
TEST_SIZE_RANDOM = 0.20
MIN_CASOS_NO_CERO = 30
EPOCHS = 50
BATCH_SIZE = 256

MIN_REGISTROS_RANKING = 2
MIN_TRABAJADORES_PROMEDIO = 10

FACTOR_CAP_LAMBDA = 2.0
CAP_MINIMO = 1.0
CAP_MAXIMO_GLOBAL = 500.0


# ============================================================
# FUNCIONES DE SALIDA Y TRAZABILIDAD
# ============================================================

def log_message(message):
    """Imprime mensajes en consola y los guarda en un archivo de log."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def export_excel(dataframe, filename, index=False):
    """Exporta DataFrames a la subcarpeta de Excel."""
    output_path = EXCEL_DIR / filename
    dataframe.to_excel(output_path, index=index)
    return output_path


def reset_execution_log():
    """Reinicia el archivo de log en cada ejecución completa de la herramienta."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Herramienta IA Riesgo Laboral V7 - Resumen de ejecución\n")
        f.write("=" * 70 + "\n")




# ============================================================
# LECTURA ROBUSTA
# ============================================================

def read_csv_robust(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin1")


# ============================================================
# CARGA Y LIMPIEZA
# ============================================================

def load_and_clean():
    df = read_csv_robust(FILE_PATH)
    df.columns = df.columns.str.strip()

    needed = CAT_COLS + NUM_COLS_BASE + TARGETS
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas en la base: {missing}")

    for col in NUM_COLS_BASE + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=CAT_COLS + NUM_COLS_BASE).copy()

    for col in TARGETS:
        df[col] = df[col].fillna(0).clip(lower=0)

    df["TRABAJADORES"] = (df["RELA_DEP"] + df["RELA_INDEP"]).clip(lower=1)
    df["LOG_TRABAJADORES"] = np.log1p(df["TRABAJADORES"])

    df["PERIODO"] = (
        df["AÑO_DE_INFORME"].astype(int).astype(str)
        + "-"
        + df["MES_DE_INFORME"].astype(int).astype(str).str.zfill(2)
    )

    return df


# ============================================================
# REPORTE DE CALIDAD
# ============================================================

def generate_quality_report(df):
    rows = []

    for col in df.columns:
        row = {
            "columna": col,
            "tipo": str(df[col].dtype),
            "faltantes": df[col].isna().sum(),
            "faltantes_%": round(df[col].isna().mean() * 100, 4),
            "valores_unicos": df[col].nunique(dropna=True)
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            row["min"] = df[col].min()
            row["max"] = df[col].max()
            row["media"] = df[col].mean()
            row["ceros_%"] = round((df[col] == 0).mean() * 100, 4)
            row["negativos"] = (df[col] < 0).sum()
        else:
            row["min"] = ""
            row["max"] = ""
            row["media"] = ""
            row["ceros_%"] = ""
            row["negativos"] = ""

        rows.append(row)

    quality = pd.DataFrame(rows)

    resumen_targets = []
    for t in TARGETS:
        resumen_targets.append({
            "variable": t,
            "media": df[t].mean(),
            "maximo": df[t].max(),
            "percentil_99": df[t].quantile(0.99),
            "percentil_999": df[t].quantile(0.999),
            "ceros_%": round((df[t] == 0).mean() * 100, 4),
            "casos_no_cero": int((df[t] > 0).sum())
        })

    resumen_targets = pd.DataFrame(resumen_targets)

    quality.to_excel(EXCEL_DIR / "01_reporte_calidad_datos_detalle.xlsx", index=False)
    resumen_targets.to_excel(EXCEL_DIR / "02_resumen_variables_objetivo.xlsx", index=False)

    return quality, resumen_targets


# ============================================================
# CODIFICACIÓN
# ============================================================

def encode_categoricals(df):
    cardinalities = {}

    for col in CAT_COLS:
        df[col] = df[col].astype(str).fillna("SIN_DATO")
        categories = pd.Categorical(df[col])
        df[col + "_CODE"] = categories.codes + 1
        cardinalities[col] = df[col + "_CODE"].max() + 1

    return df, cardinalities


def random_split_80_20(df):
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=TEST_SIZE_RANDOM,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_idx = df.index.isin(train_indices)
    test_idx = df.index.isin(test_indices)

    return train_idx, test_idx, "Aleatorio reproducible: 80% entrenamiento y 20% prueba"


def build_inputs(df, scaler=None, fit_scaler=False):
    num_cols = NUM_COLS_BASE + ["TRABAJADORES", "LOG_TRABAJADORES"]

    X_num = df[num_cols].copy()

    if fit_scaler:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = scaler.transform(X_num)

    inputs = {"num_input": X_num_scaled.astype("float32")}

    for col in CAT_COLS:
        inputs[col + "_input"] = df[col + "_CODE"].values.astype("int32")

    return inputs, scaler


# ============================================================
# MODELO ESTABLE
# ============================================================

def build_poisson_embedding_model(cardinalities, n_num_features):
    inputs = []
    embeddings = []

    num_input = keras.Input(shape=(n_num_features,), name="num_input")
    inputs.append(num_input)

    for col in CAT_COLS:
        inp = keras.Input(shape=(1,), name=col + "_input")
        inputs.append(inp)

        card = cardinalities[col]
        emb_dim = min(50, max(2, int(np.sqrt(card))))

        emb = layers.Embedding(
            input_dim=card + 1,
            output_dim=emb_dim,
            name=f"emb_{col}"
        )(inp)

        emb = layers.Flatten()(emb)
        embeddings.append(emb)

    x = layers.Concatenate()(embeddings + [num_input])

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.10)(x)

    raw_output = layers.Dense(1, name="raw_lambda_output")(x)

    output = layers.Activation(
        lambda z: tf.nn.softplus(z) + 1e-8,
        name="lambda_output"
    )(raw_output)

    model = keras.Model(inputs=inputs, outputs=output)

    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Poisson(),
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE")]
    )

    return model


# ============================================================
# CONTROL DE LAMBDAS
# ============================================================

def calculate_lambda_cap(y_train):
    y_train = np.asarray(y_train)

    p999 = np.quantile(y_train, 0.999)
    max_real = np.max(y_train)
    media = np.mean(y_train)
    desviacion = np.std(y_train)

    cap = max(
        CAP_MINIMO,
        p999 * FACTOR_CAP_LAMBDA,
        media + 10 * desviacion
    )

    cap = min(
        cap,
        max_real * FACTOR_CAP_LAMBDA if max_real > 0 else cap,
        CAP_MAXIMO_GLOBAL
    )

    cap = max(cap, CAP_MINIMO)

    return float(cap)


def apply_lambda_cap(pred, cap):
    return np.clip(pred, 1e-8, cap)


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_goodness_metrics(y_true, y_pred, target):
    y_pred_safe = np.clip(y_pred, 1e-8, None)

    mae = mean_absolute_error(y_true, y_pred_safe)
    mse = mean_squared_error(y_true, y_pred_safe)
    rmse = np.sqrt(mse)

    try:
        poisson_dev = mean_poisson_deviance(y_true, y_pred_safe)
    except ValueError:
        poisson_dev = np.nan

    try:
        r2 = r2_score(y_true, y_pred_safe)
    except ValueError:
        r2 = np.nan

    try:
        explained_var = explained_variance_score(y_true, y_pred_safe)
    except ValueError:
        explained_var = np.nan

    if np.std(y_true) > 0 and np.std(y_pred_safe) > 0:
        corr = np.corrcoef(y_true, y_pred_safe)[0, 1]
    else:
        corr = np.nan

    media_real = np.mean(y_true)
    media_pred = np.mean(y_pred_safe)
    error_media_abs = abs(media_real - media_pred)
    error_media_pct = (error_media_abs / media_real) * 100 if media_real != 0 else np.nan

    return {
        "variable": target,
        "MAE_error_absoluto_medio": mae,
        "MSE_error_cuadratico_medio": mse,
        "RMSE_raiz_error_cuadratico_medio": rmse,
        "Mean_Poisson_Deviance": poisson_dev,
        "R2": r2,
        "Varianza_explicada": explained_var,
        "Correlacion_real_predicho": corr,
        "media_real_test": media_real,
        "media_predicha_test": media_pred,
        "error_absoluto_media": error_media_abs,
        "error_porcentual_media": error_media_pct,
        "max_real_test": np.max(y_true),
        "max_pred_test": np.max(y_pred_safe),
        "ceros_%_test": round((y_true == 0).mean() * 100, 4),
        "casos_no_cero_test": int((y_true > 0).sum())
    }


# ============================================================
# ENTRENAMIENTO
# ============================================================

def train_target_model(df, target, train_idx, test_idx, cardinalities):
    casos_no_cero = int((df[target] > 0).sum())

    if casos_no_cero < MIN_CASOS_NO_CERO:
        return None, None, {
            "variable": target,
            "estado": "No entrenada",
            "motivo": f"Casos no cero insuficientes: {casos_no_cero}",
            "casos_no_cero": casos_no_cero
        }, None, None

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    X_train, scaler = build_inputs(train_df, fit_scaler=True)
    X_test, _ = build_inputs(test_df, scaler=scaler, fit_scaler=False)

    y_train = train_df[target].values.astype("float32")
    y_test = test_df[target].values.astype("float32")

    cap_lambda = calculate_lambda_cap(y_train)

    model = build_poisson_embedding_model(
        cardinalities,
        X_train["num_input"].shape[1]
    )

    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=1
    )

    pred_test_raw = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    pred_test = apply_lambda_cap(pred_test_raw, cap_lambda)

    bondad = calculate_goodness_metrics(y_test, pred_test, target)
    bondad["estado"] = "Entrenada"
    bondad["casos_no_cero_total"] = casos_no_cero
    bondad["ceros_%_total"] = round((df[target] == 0).mean() * 100, 4)
    bondad["cap_lambda_aplicado"] = cap_lambda
    bondad["max_pred_test_sin_cap"] = np.max(pred_test_raw)

    predicciones_test = test_df[
        ["DPTO", "MPIO", "ACTIVEC", "AÑO_DE_INFORME", "MES_DE_INFORME",
         "RELA_DEP", "RELA_INDEP", "TRABAJADORES"]
    ].copy()

    predicciones_test["variable_objetivo"] = target
    predicciones_test["valor_real"] = y_test
    predicciones_test["valor_predicho"] = pred_test
    predicciones_test["valor_predicho_sin_cap"] = pred_test_raw
    predicciones_test["cap_lambda_aplicado"] = cap_lambda
    predicciones_test["error"] = predicciones_test["valor_real"] - predicciones_test["valor_predicho"]
    predicciones_test["error_absoluto"] = np.abs(predicciones_test["error"])
    predicciones_test["error_cuadratico"] = predicciones_test["error"] ** 2

    all_inputs, _ = build_inputs(df, scaler=scaler, fit_scaler=False)
    pred_all_raw = model.predict(all_inputs, batch_size=BATCH_SIZE, verbose=0).ravel()
    pred_all = apply_lambda_cap(pred_all_raw, cap_lambda)

    diagnostico = {
        "variable": target,
        "cap_lambda_aplicado": cap_lambda,
        "max_lambda_sin_cap": np.max(pred_all_raw),
        "max_lambda_con_cap": np.max(pred_all),
        "media_lambda_con_cap": np.mean(pred_all),
        "percentil_99_lambda_con_cap": np.quantile(pred_all, 0.99),
        "percentil_999_lambda_con_cap": np.quantile(pred_all, 0.999),
        "n_predicciones_afectadas_por_cap": int(np.sum(pred_all_raw > cap_lambda)),
        "porcentaje_afectado_por_cap": round(np.mean(pred_all_raw > cap_lambda) * 100, 6)
    }

    return pred_all, predicciones_test, bondad, model, diagnostico


# ============================================================
# CLASIFICACIÓN
# ============================================================

def classify_by_percentiles(series):
    p60 = series.quantile(0.60)
    p90 = series.quantile(0.90)
    p97 = series.quantile(0.97)

    def clasificar(x):
        if x <= p60:
            return "Bajo"
        elif x <= p90:
            return "Medio"
        elif x <= p97:
            return "Alto"
        else:
            return "Crítico"

    return series.apply(clasificar), p60, p90, p97


# ============================================================
# EVALUACIÓN / CALIBRACIÓN
# ============================================================

def evaluate_by_risk_level(df):
    resumen = df.groupby("nivel_riesgo").agg(
        cantidad_registros=("nivel_riesgo", "size"),
        indice_promedio=("indice_riesgo_compuesto", "mean"),
        tasa_promedio=("tasa_riesgo_compuesto", "mean"),
        trabajadores_promedio=("TRABAJADORES", "mean"),
        accidentes_reales_promedio=("PRESUACCIDETRASUCE", "mean"),
        accidentes_predichos_promedio=("lambda_PRESUACCIDETRASUCE", "mean"),
        accidentes_reales_total=("PRESUACCIDETRASUCE", "sum"),
        accidentes_predichos_total=("lambda_PRESUACCIDETRASUCE", "sum"),
        muertes_reales_total=("MUERTES_REPOR_AT", "sum"),
        muertes_predichas_total=("lambda_MUERTES_REPOR_AT", "sum"),
        invalidez_at_total=("NUEVAPENSIOINVA_R_AT", "sum"),
        invalidez_at_predicha_total=("lambda_NUEVAPENSIOINVA_R_AT", "sum"),
        incapacidad_at_total=("INCAPERMAPARCIAR_AT", "sum"),
        incapacidad_at_predicha_total=("lambda_INCAPERMAPARCIAR_AT", "sum")
    ).reset_index()

    resumen["porcentaje_registros"] = (
        resumen["cantidad_registros"] / len(df) * 100
    ).round(4)

    resumen["error_abs_accidentes_promedio"] = (
        resumen["accidentes_reales_promedio"]
        - resumen["accidentes_predichos_promedio"]
    ).abs()

    return resumen


def generate_calibration_by_deciles(df):
    temp = df.copy()

    temp["decil_riesgo"] = pd.qcut(
        temp["indice_riesgo_compuesto"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)]
    )

    calibracion = temp.groupby("decil_riesgo", observed=False).agg(
        cantidad_registros=("decil_riesgo", "size"),
        indice_promedio=("indice_riesgo_compuesto", "mean"),
        accidentes_reales_promedio=("PRESUACCIDETRASUCE", "mean"),
        accidentes_predichos_promedio=("lambda_PRESUACCIDETRASUCE", "mean"),
        accidentes_reales_total=("PRESUACCIDETRASUCE", "sum"),
        accidentes_predichos_total=("lambda_PRESUACCIDETRASUCE", "sum")
    ).reset_index()

    calibracion["error_abs_promedio"] = (
        calibracion["accidentes_reales_promedio"]
        - calibracion["accidentes_predichos_promedio"]
    ).abs()

    return calibracion


# ============================================================
# ÍNDICE COMPUESTO NORMALIZADO
# ============================================================

def calculate_composite_index(df):
    escalas = []

    df["indice_riesgo_compuesto"] = 0

    for target in TARGETS:
        lambda_col = f"lambda_{target}"
        escala = max(df[lambda_col].quantile(0.99), 1.0)

        norm_col = f"lambda_norm_{target}"
        df[norm_col] = df[lambda_col] / escala

        peso = PESOS_SEVERIDAD[target]
        df["indice_riesgo_compuesto"] += peso * df[norm_col]

        escalas.append({
            "variable": target,
            "escala_p99_usada": escala,
            "peso_severidad": peso,
            "max_lambda": df[lambda_col].max(),
            "max_lambda_norm": df[norm_col].max()
        })

    pd.DataFrame(escalas).to_excel(
        EXCEL_DIR / "17_escalas_indice_compuesto.xlsx",
        index=False
    )

    return df


# ============================================================
# RANKINGS
# ============================================================

def generate_rankings(df):
    ranking = df.groupby(["DPTO", "MPIO", "ACTIVEC"]).agg(
        riesgo_accidentes_total=("lambda_PRESUACCIDETRASUCE", "sum"),
        riesgo_accidentes_promedio=("lambda_PRESUACCIDETRASUCE", "mean"),
        prob_accidente_promedio=("prob_PRESUACCIDETRASUCE", "mean"),
        indice_compuesto_total=("indice_riesgo_compuesto", "sum"),
        indice_compuesto_promedio=("indice_riesgo_compuesto", "mean"),
        trabajadores_promedio=("TRABAJADORES", "mean"),
        trabajadores_total=("TRABAJADORES", "sum"),
        registros=("indice_riesgo_compuesto", "size"),
        accidentes_reales=("PRESUACCIDETRASUCE", "sum"),
        muertes_reales=("MUERTES_REPOR_AT", "sum"),
        invalidez_at_real=("NUEVAPENSIOINVA_R_AT", "sum"),
        incapacidad_at_real=("INCAPERMAPARCIAR_AT", "sum")
    ).reset_index()

    ranking["tasa_indice_por_1000_trab"] = (
        ranking["indice_compuesto_total"] / ranking["trabajadores_total"].clip(lower=1)
    ) * 1000

    ranking_total = ranking.sort_values("indice_compuesto_total", ascending=False)

    ranking_filtrado = ranking[
        (ranking["registros"] >= MIN_REGISTROS_RANKING)
        & (ranking["trabajadores_promedio"] >= MIN_TRABAJADORES_PROMEDIO)
    ].copy()

    ranking_tasa = ranking_filtrado.sort_values(
        "tasa_indice_por_1000_trab",
        ascending=False
    )

    ranking_total.to_excel(EXCEL_DIR / "06_ranking_multicriterio_total.xlsx", index=False)
    ranking_tasa.to_excel(EXCEL_DIR / "07_ranking_multicriterio_tasa_filtrado.xlsx", index=False)

    ranking_total.head(100).to_excel(EXCEL_DIR / "08_top100_riesgo_total.xlsx", index=False)
    ranking_tasa.head(100).to_excel(EXCEL_DIR / "09_top100_riesgo_tasa_filtrado.xlsx", index=False)

    pd.DataFrame([{
        "min_registros_ranking_tasa": MIN_REGISTROS_RANKING,
        "min_trabajadores_promedio": MIN_TRABAJADORES_PROMEDIO,
        "proposito": "Evitar rankings por tasa inflados por segmentos con pocos registros o pocos trabajadores."
    }]).to_excel(EXCEL_DIR / "14_filtros_ranking_tasa.xlsx", index=False)

    return ranking_total, ranking_tasa


# ============================================================
# GRÁFICAS
# ============================================================

def generar_graficas(df, ranking_total, ranking_tasa, evaluacion_niveles, calibracion):
    distrib = df["nivel_riesgo"].value_counts().reindex(
        ["Bajo", "Medio", "Alto", "Crítico"]
    )

    plt.figure(figsize=(8, 5))
    distrib.plot(kind="bar")
    plt.title("Distribución de registros por nivel de riesgo")
    plt.xlabel("Nivel de riesgo")
    plt.ylabel("Cantidad de registros")
    plt.tight_layout()
    plt.savefig(GRAFICAS_DIR / "01_distribucion_niveles_riesgo.png", dpi=300)
    plt.close()

    top_total = ranking_total.head(20).copy()
    top_total["segmento"] = (
        top_total["DPTO"].astype(str) + " - "
        + top_total["MPIO"].astype(str) + " - "
        + top_total["ACTIVEC"].astype(str)
    )

    plt.figure(figsize=(12, 7))
    plt.barh(top_total["segmento"], top_total["indice_compuesto_total"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 segmentos por riesgo compuesto total")
    plt.xlabel("Índice compuesto total normalizado")
    plt.ylabel("Segmento")
    plt.tight_layout()
    plt.savefig(GRAFICAS_DIR / "02_top20_riesgo_total.png", dpi=300)
    plt.close()

    if not ranking_tasa.empty:
        top_tasa = ranking_tasa.head(20).copy()
        top_tasa["segmento"] = (
            top_tasa["DPTO"].astype(str) + " - "
            + top_tasa["MPIO"].astype(str) + " - "
            + top_tasa["ACTIVEC"].astype(str)
        )

        plt.figure(figsize=(12, 7))
        plt.barh(top_tasa["segmento"], top_tasa["tasa_indice_por_1000_trab"])
        plt.gca().invert_yaxis()
        plt.title("Top 20 segmentos por tasa de riesgo filtrada")
        plt.xlabel("Tasa de índice por 1000 trabajadores")
        plt.ylabel("Segmento")
        plt.tight_layout()
        plt.savefig(GRAFICAS_DIR / "03_top20_riesgo_tasa_filtrada.png", dpi=300)
        plt.close()
    else:
        pd.DataFrame([{
            "mensaje": "No se generó gráfica de tasa porque ningún segmento cumplió los filtros definidos."
        }]).to_excel(EXCEL_DIR / "18_ranking_tasa_vacio.xlsx", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(calibracion["decil_riesgo"], calibracion["accidentes_reales_promedio"], marker="o", label="Real")
    plt.plot(calibracion["decil_riesgo"], calibracion["accidentes_predichos_promedio"], marker="o", label="Predicho")
    plt.title("Calibración del modelo por deciles de riesgo")
    plt.xlabel("Decil de riesgo")
    plt.ylabel("Accidentes promedio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICAS_DIR / "04_calibracion_deciles.png", dpi=300)
    plt.close()

    orden = ["Bajo", "Medio", "Alto", "Crítico"]
    evaluacion_niveles["nivel_riesgo"] = pd.Categorical(
        evaluacion_niveles["nivel_riesgo"],
        categories=orden,
        ordered=True
    )
    evaluacion_niveles = evaluacion_niveles.sort_values("nivel_riesgo")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(evaluacion_niveles))
    width = 0.35

    plt.bar(x - width / 2, evaluacion_niveles["accidentes_reales_promedio"], width, label="Real")
    plt.bar(x + width / 2, evaluacion_niveles["accidentes_predichos_promedio"], width, label="Predicho")

    plt.xticks(x, evaluacion_niveles["nivel_riesgo"])
    plt.title("Accidentes reales vs predichos por nivel de riesgo")
    plt.xlabel("Nivel de riesgo")
    plt.ylabel("Accidentes promedio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICAS_DIR / "05_real_vs_predicho_niveles.png", dpi=300)
    plt.close()


# ============================================================
# DASHBOARD HTML AUTOMÁTICO
# ============================================================

def generar_dashboard_html(df, metricas_df, ranking_total, ranking_tasa, evaluacion_niveles, calibracion):
    total_registros = len(df)
    total_municipios = df["MPIO"].nunique()
    total_actividades = df["ACTIVEC"].nunique()
    total_departamentos = df["DPTO"].nunique()

    riesgo_counts = df["nivel_riesgo"].value_counts()
    riesgo_critico = int(riesgo_counts.get("Crítico", 0))
    riesgo_alto = int(riesgo_counts.get("Alto", 0))

    top_total = ranking_total.head(5)[
        ["DPTO", "MPIO", "ACTIVEC", "indice_compuesto_total"]
    ]

    if not ranking_tasa.empty:
        top_tasa = ranking_tasa.head(5)[
            ["DPTO", "MPIO", "ACTIVEC", "tasa_indice_por_1000_trab"]
        ]
        top_tasa_html = top_tasa.to_html(index=False, classes="tabla")
    else:
        top_tasa_html = "<p>No hubo registros suficientes para ranking por tasa.</p>"

    metricas_html = metricas_df.to_html(index=False, classes="tabla")
    top_total_html = top_total.to_html(index=False, classes="tabla")
    niveles_html = evaluacion_niveles.to_html(index=False, classes="tabla")
    calibracion_html = calibracion.to_html(index=False, classes="tabla")

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Dashboard SST PredictIA</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
                background-color: #f4f6f8;
                color: #222;
            }}
            h1, h2 {{
                color: #12355b;
            }}
            .card {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            .kpi-container {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
            }}
            .kpi {{
                background: #eaf2f8;
                padding: 18px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
            }}
            .kpi span {{
                display: block;
                font-size: 26px;
                color: #0b5394;
                margin-top: 8px;
            }}
            .tabla {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .tabla th {{
                background: #12355b;
                color: white;
                padding: 8px;
            }}
            .tabla td {{
                border: 1px solid #ddd;
                padding: 7px;
            }}
            img {{
                max-width: 100%;
                border-radius: 10px;
                border: 1px solid #ddd;
                margin-top: 10px;
            }}
            .recomendacion {{
                background: #fff3cd;
                padding: 15px;
                border-left: 5px solid #f0ad4e;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>

        <h1>Dashboard SST PredictIA</h1>
        <p><b>Herramienta de inteligencia artificial para predicción, clasificación y priorización del riesgo laboral.</b></p>

        <div class="card">
            <h2>1. Resumen ejecutivo</h2>
            <div class="kpi-container">
                <div class="kpi">Registros analizados<span>{total_registros:,}</span></div>
                <div class="kpi">Departamentos<span>{total_departamentos}</span></div>
                <div class="kpi">Municipios<span>{total_municipios}</span></div>
                <div class="kpi">Actividades económicas<span>{total_actividades}</span></div>
            </div>
        </div>

        <div class="card">
            <h2>2. Alertas de riesgo</h2>
            <div class="kpi-container">
                <div class="kpi">Riesgo alto<span>{riesgo_alto:,}</span></div>
                <div class="kpi">Riesgo crítico<span>{riesgo_critico:,}</span></div>
            </div>
            <div class="recomendacion">
                <b>Recomendación:</b> priorizar los segmentos clasificados como riesgo crítico y alto para campañas preventivas, inspecciones, capacitaciones y seguimiento en SST.
            </div>
        </div>

        <div class="card">
            <h2>3. Distribución de niveles de riesgo</h2>
            <img src="../graficas/01_distribucion_niveles_riesgo.png">
        </div>

        <div class="card">
            <h2>4. Top 20 segmentos por riesgo compuesto total</h2>
            <img src="../graficas/02_top20_riesgo_total.png">
            <h3>Top 5</h3>
            {top_total_html}
        </div>

        <div class="card">
            <h2>5. Top 20 segmentos por tasa de riesgo filtrada</h2>
            <img src="../graficas/03_top20_riesgo_tasa_filtrada.png">
            <h3>Top 5</h3>
            {top_tasa_html}
        </div>

        <div class="card">
            <h2>6. Calibración del modelo</h2>
            <img src="../graficas/04_calibracion_deciles.png">
            <h3>Tabla de calibración</h3>
            {calibracion_html}
        </div>

        <div class="card">
            <h2>7. Accidentes reales vs predichos por nivel de riesgo</h2>
            <img src="../graficas/05_real_vs_predicho_niveles.png">
            <h3>Evaluación por nivel</h3>
            {niveles_html}
        </div>

        <div class="card">
            <h2>8. Métricas de bondad del modelo</h2>
            <p>Estas métricas permiten evaluar qué tan bien predice la red neuronal en el 20% de prueba.</p>
            {metricas_html}
        </div>

        <div class="card">
            <h2>9. Conclusión automática</h2>
            <p>
            La herramienta permite analizar datos históricos de accidentalidad laboral,
            estimar el riesgo esperado, clasificar los registros en niveles de riesgo,
            validar la bondad del modelo y generar rankings de priorización territorial
            y sectorial para apoyar la toma de decisiones preventivas en SST.
            </p>
        </div>

    </body>
    </html>
    """

    dashboard_path = DASHBOARD_DIR / "Dashboard_SST_PredictIA.html"

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)

    log_message(f"Dashboard generado: {dashboard_path.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():
    reset_execution_log()
    log_message("Cargando y limpiando datos...")
    df = load_and_clean()

    log_message(f"Registros cargados: {len(df):,}")
    log_message(f"Periodos encontrados: {sorted(df['PERIODO'].unique())}")

    log_message("Generando reporte de calidad...")
    generate_quality_report(df)

    log_message("Codificando variables categóricas...")
    df, cardinalities = encode_categoricals(df)

    train_idx, test_idx, tipo_split = random_split_80_20(df)

    pd.DataFrame([{
        "tipo_split": tipo_split,
        "porcentaje_entrenamiento": 80,
        "porcentaje_prueba": 20,
        "registros_entrenamiento": int(train_idx.sum()),
        "registros_prueba": int(test_idx.sum()),
        "random_state": RANDOM_STATE
    }]).to_excel(EXCEL_DIR / "00_configuracion_validacion_80_20.xlsx", index=False)

    metricas_bondad = []
    predicciones_test_list = []
    diagnosticos_lambdas = []

    for target in TARGETS:
        log_message(f"\nEntrenando modelo para: {target}")

        pred_all, pred_test_df, bondad, model, diagnostico = train_target_model(
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            cardinalities=cardinalities
        )

        metricas_bondad.append(bondad)

        if diagnostico is not None:
            diagnosticos_lambdas.append(diagnostico)

        if pred_test_df is not None:
            predicciones_test_list.append(pred_test_df)

        if pred_all is not None:
            df[f"lambda_{target}"] = pred_all
            df[f"prob_{target}"] = 1 - np.exp(-pred_all)
        else:
            df[f"lambda_{target}"] = 0
            df[f"prob_{target}"] = 0

    metricas_df = pd.DataFrame(metricas_bondad)

    metricas_df.to_excel(
        EXCEL_DIR / "11_metricas_bondad_modelo.xlsx",
        index=False
    )

    metricas_df.to_excel(
        EXCEL_DIR / "03_metricas_modelos_por_variable.xlsx",
        index=False
    )

    if diagnosticos_lambdas:
        pd.DataFrame(diagnosticos_lambdas).to_excel(
            EXCEL_DIR / "16_revision_lambdas_maximas.xlsx",
            index=False
        )

    if predicciones_test_list:
        pd.concat(predicciones_test_list, ignore_index=True).to_excel(
            EXCEL_DIR / "13_predicciones_test_20_por_ciento.xlsx",
            index=False
        )

    log_message("Calculando índice compuesto normalizado...")
    df = calculate_composite_index(df)

    df["nivel_riesgo"], p60, p90, p97 = classify_by_percentiles(
        df["indice_riesgo_compuesto"]
    )

    df["tasa_riesgo_compuesto"] = (
        df["indice_riesgo_compuesto"] / df["TRABAJADORES"].clip(lower=1)
    ) * 1000

    df["nivel_riesgo"].value_counts().reset_index().rename(
        columns={"index": "nivel_riesgo", "nivel_riesgo": "cantidad_registros"}
    ).to_excel(EXCEL_DIR / "04_distribucion_niveles_riesgo.xlsx", index=False)

    pd.DataFrame([{
        "p60_bajo_medio": p60,
        "p90_medio_alto": p90,
        "p97_alto_critico": p97
    }]).to_excel(EXCEL_DIR / "05_umbrales_clasificacion_riesgo.xlsx", index=False)

    evaluacion_niveles = evaluate_by_risk_level(df)
    evaluacion_niveles.to_excel(
        EXCEL_DIR / "12_evaluacion_por_nivel_riesgo.xlsx",
        index=False
    )

    calibracion = generate_calibration_by_deciles(df)
    calibracion.to_excel(
        EXCEL_DIR / "15_calibracion_por_deciles.xlsx",
        index=False
    )

    ranking_total, ranking_tasa = generate_rankings(df)

    generar_graficas(
        df=df,
        ranking_total=ranking_total,
        ranking_tasa=ranking_tasa,
        evaluacion_niveles=evaluacion_niveles,
        calibracion=calibracion
    )

    log_message("Generando dashboard HTML automático...")
    generar_dashboard_html(
        df=df,
        metricas_df=metricas_df,
        ranking_total=ranking_total,
        ranking_tasa=ranking_tasa,
        evaluacion_niveles=evaluacion_niveles,
        calibracion=calibracion
    )

    df.to_excel(
        EXCEL_DIR / "10_predicciones_completas_herramienta_sst.xlsx",
        index=False
    )

    log_message("\nProceso finalizado correctamente.")
    log_message(f"Resultados guardados en la carpeta: {OUTPUT_DIR.resolve()}")
    log_message(f"Archivos Excel: {EXCEL_DIR.resolve()}")
    log_message(f"Gráficas: {GRAFICAS_DIR.resolve()}")
    log_message(f"Dashboard: {DASHBOARD_DIR.resolve()}")
    log_message(f"Log de ejecución: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()