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

## Estructura recomendada del proyecto

```text
CODE/
│
├── README.md
├── requirements.txt
├── .gitignore
├── Herramienta_IA_Riesgo_Laboral_V7.ipynb
├── Herramienta_IA_Riesgo_Laboral_V7_actualizada.py
│
├── ARCHIVOS PROYECTO/
│   ├── Base de datos limpia postunificacion/
│   ├── Bases de datos adquiridas preunificacion/
│   └── Pruebas antiguas/
│
├── Legacy code/
│
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

