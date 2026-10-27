# Auditoría técnica y académica — Herramienta de IA para riesgo laboral V7

**Base evaluada:** reconstrucción Positiva 2025–2026 (165.313 registros).  
**Periodos:** septiembre de 2025; noviembre de 2025; febrero de 2026.  
**Origen:** archivo de resultados `resultados_herramienta_sst_v7_dashboard.rar` y CSV reconstruido usado para reproducir los análisis complementarios.  
**Alcance:** revisión documental, cálculos descriptivos y recomputaciones sobre resultados exportados. No se reentrenaron redes neuronales.

## 1. Estado de la ejecución

La ejecución finalizó correctamente. Se utilizaron 132.250 observaciones para entrenamiento y 33.063 para prueba con división aleatoria 80/20 y semilla 42. Se entrenaron los modelos de los seis desenlaces y se generaron métricas, predicciones, clasificaciones, rankings y dashboard.

Se identificaron 34 departamentos, 1.052 municipios y 991 actividades económicas en el reporte de calidad. En total, hay 66.704 agrupaciones distintas de departamento–municipio–actividad.

## 2. Métricas sobre el conjunto de prueba

| Desenlace | Observaciones con valor positivo (base completa) | MAE | RMSE | R² | Límite superior de lambda |
|---|---:|---:|---:|---:|---:|
| Presuntos accidentes de trabajo | 9.321 | 0,156958 | 1,902130 | 0,400872 | 34 |
| Muertes reportadas por AT | 6.069 | 0,122588 | 1,890312 | 0,260575 | 24 |
| Nueva pensión de invalidez AT | 6.065 | 0,122011 | 1,891302 | 0,259789 | 24 |
| Nueva pensión de invalidez EL | 53 | 0,000681 | 0,018240 | -0,000312 | 1 |
| Incapacidad permanente parcial AT | 139 | 0,001705 | 0,029585 | -0,158433 | 1 |
| Incapacidad permanente parcial EL | 60 | 0,000713 | 0,029322 | 0,082874 | 1 |

Las magnitudes pequeñas de MAE en desenlaces raros deben interpretarse junto con la proporción de ceros y las métricas de bondad; no constituyen evidencia suficiente de capacidad predictiva satisfactoria.

### Comparación sobre el desenlace principal

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Promedio histórico (DummyRegressor) | 0,310174 | 2,457452 | ≈0 |
| Regresión Poisson | 0,271984 | 2,253017 | 0,159441 |
| Random Forest | **0,156171** | **1,858290** | **0,428171** |
| HistGradientBoosting | 0,253647 | 2,136336 | 0,244249 |
| MLP V7 | 0,156958 | 1,902130 | 0,400872 |

Random Forest obtiene los mejores valores de las principales métricas, pero no se aplicaron intervalos de confianza o pruebas de significación para demostrar superioridad estadística.

## 3. Control de predicciones

Para accidentes el control superior de lambda fue de 34 eventos por registro; se acotaron 58 predicciones de 165.313 (0,035085%). El máximo previo al acotamiento era 273,434. En el Top 100 del ranking total, 15 agrupaciones presentan exactamente 102 accidentes esperados acumulados: tres registros con predicción acotada a 34 cada uno. El acotamiento debe evaluarse especialmente en la cola de la distribución.

## 4. Clasificación

| Nivel | Registros | Aproximación porcentual |
|---|---:|---:|
| Bajo | 99.188 | 60,00 % |
| Medio | 49.593 | 30,00 % |
| Alto | 11.572 | 7,00 % |
| Crítico | 4.960 | 3,00 % |

Percentiles sobre índice compuesto: P60 = 0,026520; P90 = 0,599609; P97 = 3,227610. Estas clases son relativas a la distribución analizada, no niveles normativos de peligrosidad.

## 5. Comparación complementaria de rankings

Se compararon dos criterios ya disponibles en los archivos exportados: ordenamiento por índice compuesto vs. ordenamiento por predicción de accidentes; no se realizó todavía un análisis formal modificando los pesos de severidad.

- Rankings por acumulado, 66.704 segmentos: Top 100 compartido = 94 de 100, Top 20 compartido = 18 de 20; correlación de Spearman entre puntuaciones ≈ 0,9634.
- Rankings relativos filtrados, 18.581 segmentos: Top 100 compartido = 51 de 100, Top 20 compartido = 13 de 20; correlación de Spearman ≈ 0,8756.
- En el Top 100 relativo del índice compuesto, 33 agrupaciones no registraron accidentes observados durante los meses incluidos; ello no equivale automáticamente a que su riesgo predictivo sea cero, pero requiere cautela en la interpretación.

El ranking relativo es más sensible al criterio empleado que el ranking por acumulado.

## 6. Condiciones de la información y limitaciones de validación

- Los tres periodos son discontinuos; no constituyen una serie mensual completa.
- En la base reconstruida, 36.584 registros contienen suma de trabajadores igual a cero; cinco de ellos tienen al menos un desenlace laboral positivo. El código sustituye cero por uno en la variable derivada `TRABAJADORES`.
- Noviembre de 2025 y febrero de 2026 muestran coincidencia exacta entre las columnas `PRESUACCIDETRASUCE`, `MUERTES_REPOR_AT` y `NUEVAPENSIOINVA_R_AT`. Se conservaron por corresponder a los archivos proporcionados como fuente; la publicación oficial no demuestra por sí misma validez semántica de todos los valores.
- Totales observados en la base reconstruida: 26.530 presuntos accidentes, 16.707 muertes reportadas por AT y 16.702 nuevas pensiones de invalidez por AT. Por esa relación extraordinaria, la interpretación de las variables graves y del índice compuesto requiere advertencia específica.
- La partición aleatoria deja 28.997 de los 33.063 registros de prueba (87,70 %) en segmentos departamento–municipio–actividad presentes también en entrenamiento. La prueba evalúa filas no utilizadas para ajustar el modelo, pero no mide generalización a segmentos totalmente nuevos ni predicción prospectiva de meses futuros.
- No se incluye validación temporal o por grupo ni análisis de robustez de las ponderaciones de severidad en esta ejecución.

## 7. Dictamen y siguientes comprobaciones

**Estado técnico:** ejecución completa; seis modelos entrenados y reportes generados.  
**Estado metodológico:** resultados exploratorios de una validación aleatoria reproducible. No es suficiente para afirmar superioridad de la MLP ni validez predictiva prospectiva.  
**Estado académico:** usar las métricas descriptivas con sus limitaciones; no formular conclusiones definitivas de severidad ni del índice multicriterio hasta documentar sensibilidad y origen de coincidencias en los desenlaces.

Prioridades de mejora: (1) evaluación temporal y por grupos de municipios–actividades; (2) revisión del acotamiento de lambda en la cola; (3) sensibilidad del índice a pesos y desenlaces graves; (4) incorporación en la sección 5.3.5 de las cifras de reconstrucción y limpieza; (5) revisión de la elección de MLP frente a Random Forest a la luz de validaciones adicionales.

## 8. Archivos usados para la auditoría

- `logs/resumen_ejecucion_v7.txt`
- `excel/00_configuracion_validacion_80_20.xlsx`
- `excel/01_reporte_calidad_datos_detalle.xlsx`
- `excel/02_resumen_variables_objetivo.xlsx`
- `excel/03_metricas_modelos_por_variable.xlsx`
- `excel/04_distribucion_niveles_riesgo.xlsx`
- `excel/05_umbrales_clasificacion_riesgo.xlsx`
- `excel/06_ranking_multicriterio_total.xlsx`
- `excel/07_ranking_multicriterio_tasa_filtrado.xlsx`
- `excel/08_top100_riesgo_total.xlsx`
- `excel/09_top100_riesgo_tasa_filtrado.xlsx`
- `excel/12_evaluacion_por_nivel_riesgo.xlsx`
- `excel/15_calibracion_por_deciles.xlsx`
- `excel/16_revision_lambdas_maximas.xlsx`
- `excel/17_escalas_indice_compuesto.xlsx`
- `excel/19_comparacion_modelos_base_presuaccidetrasuce.xlsx`
- Archivo CSV reconstruido, para comprobación de coincidencias por periodo y solapamiento entre entrenamiento y prueba.
