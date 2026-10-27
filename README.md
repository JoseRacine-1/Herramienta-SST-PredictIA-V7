# Herramienta IA Riesgo Laboral V7

Herramienta de inteligencia artificial desarrollada en Python para analizar registros históricos de riesgos laborales, estimar valores esperados de accidentalidad y severidad, clasificar niveles de riesgo y generar rankings de priorización territorial y sectorial como apoyo a la gestión preventiva en Seguridad y Salud en el Trabajo (SST).

## Versión

**Herramienta V7**  
Esta versión corresponde a la versión definitiva utilizada en la tesis de maestría.

## Propósito

La herramienta permite transformar una base histórica de riesgos laborales en salidas analíticas útiles para la toma de decisiones preventivas. Su objetivo no es predecir accidentes individuales ni determinar responsabilidades legales, sino estimar el riesgo laboral agregado por segmentos definidos por variables territoriales, sectoriales, temporales y organizacionales.

## Fuente de datos

La herramienta trabaja con una base CSV de riesgos laborales asociada a ARL Positiva. La base debe contener, como mínimo, las siguientes columnas:

- `DPTO`
- `MPIO`
- `ACTIVEC`
- `RELA_DEP`
- `RELA_INDEP`
- `AÑO_DE_INFORME`
- `MES_DE_INFORME`
- `PRESUACCIDETRASUCE`
- `MUERTES_REPOR_AT`
- `NUEVAPENSIOINVA_R_AT`
- `NUEVAPENSIOINVA_R_EL`
- `INCAPERMAPARCIAR_AT`
- `INCAPERMAPARCIAR_EL`

> Nota: por criterios de confidencialidad y control académico, no se recomienda publicar la base completa en repositorios públicos. En caso de subir el proyecto a GitHub, se sugiere incluir únicamente una muestra anonimizada o instrucciones para ubicar la base localmente.

## 1. Estructura del proyecto

```text
CODE/
├── README.md                              # Información general de la herramienta
├── Herramienta_IA_Riesgo_Laboral_V7_actualizada.py
├── archivos_proyecto/
│   └── base_limpia_postunificacion/
│       ├── README.md                      # Este documento (sí se sube a GitHub)
│       └── <archivo_operativo>.csv         # Base local (no se sube por defecto)
└── resultados_herramienta_sst_v7_dashboard/
    ├── excel/
    ├── graficas/
    ├── dashboard/
    └── logs/
```

## Instalación

1. Crear o activar un entorno de Python.
2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

3. Verificar que la ruta `FILE_PATH` dentro del código apunte al archivo CSV definitivo.

## Ejecución

Desde Jupyter Notebook, ejecutar todas las celdas del archivo:

```text
Herramienta_IA_Riesgo_Laboral_V7.ipynb
```

O desde consola, si se utiliza la versión `.py`:

```bash
python Herramienta_IA_Riesgo_Laboral_V7_actualizada.py
```

## Metodología implementada

La herramienta ejecuta el siguiente flujo:

1. Carga y lectura robusta del archivo CSV.
2. Limpieza y depuración de datos.
3. Evaluación de calidad de datos.
4. Codificación de variables categóricas.
5. Construcción de variables derivadas.
6. División 80/20 para entrenamiento y prueba.
7. Entrenamiento de modelos neuronales por variable objetivo.
8. Evaluación de bondad del modelo.
9. Generación de predicciones completas.
10. Construcción del índice compuesto de riesgo.
11. Clasificación en niveles bajo, medio, alto y crítico.
12. Generación de rankings por riesgo total y tasa ajustada.
13. Exportación de reportes Excel.
14. Generación de gráficas a 300 dpi.
15. Construcción de dashboard HTML.
16. Registro del log de ejecución.

## Variables objetivo

La herramienta predice las siguientes variables:

| Variable | Descripción |
|---|---|
| `PRESUACCIDETRASUCE` | Presuntos accidentes de trabajo sucedidos |
| `MUERTES_REPOR_AT` | Muertes reportadas por accidente de trabajo |
| `NUEVAPENSIOINVA_R_AT` | Nuevas pensiones de invalidez por accidente de trabajo |
| `NUEVAPENSIOINVA_R_EL` | Nuevas pensiones de invalidez por enfermedad laboral |
| `INCAPERMAPARCIAR_AT` | Incapacidades permanentes parciales por accidente de trabajo |
| `INCAPERMAPARCIAR_EL` | Incapacidades permanentes parciales por enfermedad laboral |

## Salidas generadas

La herramienta genera automáticamente una carpeta denominada:

```text
resultados_herramienta_sst_v7_dashboard/
```

Dentro de esta carpeta se crean subcarpetas para organizar los productos:

```text
excel/
graficas/
dashboard/
logs/
```

### Archivos Excel

La subcarpeta `excel/` contiene reportes de calidad, métricas, predicciones, umbrales, rankings y resultados completos.

### Gráficas

La subcarpeta `graficas/` contiene imágenes en formato PNG con resolución de 300 dpi, aptas para revisión e incorporación en el documento de tesis.

### Dashboard

La subcarpeta `dashboard/` contiene el tablero HTML automático de la herramienta.

### Logs

La subcarpeta `logs/` contiene el archivo `resumen_ejecucion_v7.txt`, utilizado para registrar la trazabilidad básica de la ejecución.

## Alcance

La herramienta permite estimar, clasificar y priorizar el riesgo laboral agregado a partir de datos históricos. Sus resultados deben interpretarse como apoyo analítico para la gestión preventiva en SST.

## Limitaciones

- No predice accidentes individuales.
- No identifica causas inmediatas de accidentes.
- No sustituye el criterio técnico de profesionales en SST.
- No determina responsabilidades legales.
- Su desempeño depende de la calidad, cobertura y estructura de la base de datos utilizada.

## Autoría y uso

Herramienta desarrollada con fines académicos en el marco de una tesis de maestría en Ingeniería Industrial.

## 2. Base documentada en la ejecución de la tesis

El conjunto reconstruido asociado a la ejecución y a los análisis de SST PredictIA V7 contiene **165.313 registros** distribuidos así:

| Periodo disponible | Registros conservados |
| --- | ---: |
| Septiembre de 2025 | 44.310 |
| Noviembre de 2025 | 60.684 |
| Febrero de 2026 | 60.319 |
| **Total** | **165.313** |

Estos son **tres meses específicos y discontinuos**: no representan todos los meses comprendidos entre septiembre de 2025 y febrero de 2026. El conjunto contiene **14 columnas originales** y registra 34 categorías en `DPTO`, 1.052 municipios en `MPIO` y 991 códigos de actividad económica en `ACTIVEC`.

La reconstrucción se obtuvo de **209.623 filas originales**: 88.620 de septiembre, 60.684 de noviembre y 60.319 de febrero. En septiembre se identificaron **44.310 pares duplicados** por combinación territorial, sectorial y temporal; se conservó de cada par la fila con trabajadores positivos, después de comprobar la coincidencia de sus seis desenlaces. En febrero se interpretaron separadores de miles presentes en campos de trabajadores. No se corrigieron arbitrariamente los valores de las seis variables objetivo.

> **Estado metodológico:** la reconstrucción generó originalmente el archivo `Base_Positiva_2025_2026_RECONSTRUIDA_PRELIMINAR_NO_ENTRENAR.csv` y un informe que lo identifica como **preliminar, no apto para entrenamiento definitivo sin verificación de los desenlaces**. La ejecución posteriormente analizada en la tesis utilizó una base de 165.313 registros, pero **la validación semántica de determinados eventos graves continúa pendiente**. Este README documenta la situación; no reemplaza la comprobación de la versión exacta del CSV utilizada en cada corrida.

## 3. Esquema de columnas de entrada

| Columna | Significado |
| --- | --- |
| `DPTO` | Departamento o categoría territorial equivalente. |
| `MPIO` | Municipio registrado. |
| `CODIGO_DE_LA_ARL` | Código de la administradora de riesgos laborales indicado en la fuente. |
| `AÑO_DE_INFORME` | Año del registro. |
| `MES_DE_INFORME` | Mes del registro. |
| `ACTIVEC` | Código de actividad económica; requiere consultar el diccionario de la fuente para asignarle una descripción. |
| `RELA_DEP` | Trabajadores dependientes reportados. |
| `RELA_INDEP` | Trabajadores independientes reportados. |
| `PRESUACCIDETRASUCE` | Presuntos accidentes de trabajo sucedidos. |
| `MUERTES_REPOR_AT` | Muertes reportadas por accidente de trabajo, según el campo de origen. |
| `NUEVAPENSIOINVA_R_AT` | Nuevas pensiones de invalidez por accidente de trabajo. |
| `NUEVAPENSIOINVA_R_EL` | Nuevas pensiones de invalidez por enfermedad laboral. |
| `INCAPERMAPARCIAR_AT` | Incapacidades permanentes parciales por accidente de trabajo. |
| `INCAPERMAPARCIAR_EL` | Incapacidades permanentes parciales por enfermedad laboral. |

En el procesamiento de V7 se generan, entre otras, las columnas derivadas `TRABAJADORES`, `LOG_TRABAJADORES` y `PERIODO`. **No deben confundirse con columnas originales del CSV.** La versión procesada de `TRABAJADORES` aplica un mínimo de una unidad para evitar denominadores iguales a cero; ese ajuste **no acredita la existencia de un trabajador real** cuando la fuente original reporta cero.

## 4. Hallazgos de calidad pendientes de aclaración

- **Registros sin trabajadores:** 36.584 observaciones tienen `RELA_DEP + RELA_INDEP = 0` en la base reconstruida; cinco de ellas presentan al menos un evento positivo.
- **Coincidencia de desenlaces:** en noviembre de 2025 y febrero de 2026 coinciden fila por fila los campos `PRESUACCIDETRASUCE`, `MUERTES_REPOR_AT` y `NUEVAPENSIOINVA_R_AT`. Son campos conceptualmente diferentes; su igualdad necesita verificación administrativa y **no autoriza interpretar automáticamente sus valores como muertes o invalideces confirmadas**.
- **Cobertura temporal limitada:** no hay una serie mensual completa para analizar estacionalidad o estabilidad predictiva sostenida.
- **Unidad agregada:** las filas no permiten identificar accidentes individuales, empresas únicas ni la cantidad de trabajadores distintos a lo largo de varios meses.

Estas alertas afectan particularmente la interpretación de los **modelos de desenlaces graves**, el **índice compuesto** y los **rankings**. No deben eliminarse del informe ni «corregirse» los eventos sin respaldo documental de la fuente.

## 5. Cómo debe utilizarse esta carpeta

1. Conservar aquí **solo un CSV operativo** de entrada y este `README.md`.
2. No guardar en esta ubicación archivos como `01_metricas_particiones.csv`, `01_resumen_sensibilidad.csv`, `03_niveles_escenarios.csv` ni exportaciones de auditorías. Ubicarlos en carpetas de documentación/validación, separadas de la base de entrada.
3. Antes de ejecutar V7, revisar el mensaje de consola **`Base de datos utilizada:`** y confirmar la ruta exacta del CSV.
4. Comprobar que el archivo contiene las columnas requeridas y la cantidad de registros prevista para la corrida que se desea reproducir. No mezclar archivos originales, reconstruidos y corregidos bajo un mismo nombre sin control de versiones.
5. Conservar una copia de seguridad inalterada de los datos, del script de reconstrucción y de los resultados utilizados en la tesis. Una nueva ejecución puede regenerar archivos dentro de `resultados_herramienta_sst_v7_dashboard/`.

**Detección automática:** la función `detectar_archivo_csv()` busca primero los nombres de archivo previstos por el código. Si no encuentra uno de ellos, intenta identificar un único CSV que incluya `positiva`, `2025` y `2026` en el nombre; si existe un solo CSV en la carpeta, también puede seleccionarlo. La selección falla cuando hay candidatos ambiguos. Por ello, el nombre que aparece en la consola es la comprobación decisiva y **no conviene confiar únicamente en la denominación de la carpeta**.

## 6. Procedencia, reconstrucción y trazabilidad

Los datos proceden de archivos históricos atribuidos a **Positiva Compañía de Seguros S. A.**, obtenidos de la fuente pública de estadísticas de riesgos laborales utilizada en la investigación. Su reconstrucción documental se realizó sin modificar los archivos originales; las operaciones principales fueron homologación de columnas, normalización de formatos numéricos y eliminación trazada de duplicaciones de septiembre.

Para reproducir o auditar el procedimiento deben conservarse, en la documentación del proyecto o en un respaldo local controlado:

- Los **tres archivos CSV originales**, sin modificaciones.
- El script **`reconstruccion_controlada_v7.py`**.
- El archivo de trazabilidad **`Trazabilidad_pares_septiembre_2025.csv`** y el informe **`Informe_reconstruccion_preliminar_V7.txt`**.
- Los scripts de auditoría, validación temporal y por grupos y análisis de sensibilidad, junto con los metadatos de sus ejecuciones.
- La versión exacta del archivo operativo y de `Herramienta_IA_Riesgo_Laboral_V7_actualizada.py` utilizada para producir los resultados de la tesis.

**Control de integridad de referencia.** El archivo de reconstrucción preliminar verificado en el entorno de análisis, `Base_Positiva_2025_2026_RECONSTRUIDA_PRELIMINAR_NO_ENTRENAR.csv`, tiene la siguiente huella SHA-256:

```text
66e2ce3410fc1e1a1a5649648bfbe45ab85e3ab377a9578dbe77a8e86ba979c4
```

Esta huella **solo identifica exactamente ese archivo reconstruido**. Si el CSV local operativo tiene otro nombre, otra codificación o cambios de contenido, su hash puede ser distinto. Para saber si es el mismo archivo byte a byte, en Windows PowerShell puede ejecutarse:

```powershell
Get-FileHash -Algorithm SHA256 ".\nombre_real_del_archivo.csv"
```

Si la huella no coincide, documentar por separado la versión ejecutada; **no afirmar que ambas bases son idénticas únicamente porque tienen 165.313 filas**.

## 7. Publicación en GitHub

Se recomienda **versionar este README**, el código de la herramienta y los scripts de reconstrucción y validación, pero **no publicar automáticamente el CSV completo**. Antes de distribuir bases o registros detallados, verificar la licencia y condiciones de redistribución de la fuente, la posible presencia de información identificable y la pertinencia académica de su publicación.

En el `.gitignore` de la raíz del repositorio puede utilizarse esta regla, siempre que la carpeta conserve el nombre indicado:

```gitignore
/archivos_proyecto/base_limpia_postunificacion/*
!/archivos_proyecto/base_limpia_postunificacion/README.md
```

Esta regla no elimina archivos del computador ni retira de Git archivos que ya hubieran sido versionados previamente. Si el CSV ya está siendo seguido por Git, debe retirarse **solo del índice** después de confirmar que se dispone de un respaldo y de revisar el historial de publicación.

## 8. Interpretación y uso responsable

Los archivos de esta carpeta se utilizan para investigar **estimaciones y prioridades de riesgo laboral agregado**. Las predicciones no deben tratarse como diagnósticos individuales, demostraciones de causalidad, mediciones normativas del nivel de peligro o decisiones autónomas de intervención. El índice compuesto utiliza ponderaciones definidas en la investigación; la puntuación relativa por 1.000 trabajadores no es una tasa epidemiológica validada.

**Estado de la investigación:** ejecución y análisis de la Herramienta V7 documentados con 165.313 registros; pendientes de validación sustantiva determinadas variables de severidad y la trazabilidad exacta entre el archivo reconstruido preliminar y el CSV operativo conservado por el autor.

Para la instalación, ejecución y descripción general del modelo, consultar el `README.md` de la raíz del proyecto.
