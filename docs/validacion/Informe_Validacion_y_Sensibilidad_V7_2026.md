# Evaluación complementaria de robustez y generalización — Herramienta V7

**Fecha de análisis:** 23 de octubre de 2026.  
**Base:** reconstrucción preliminar con 165.313 registros, periodos 2025-09, 2025-11 y 2026-02.  
**Alcance:** análisis de sensibilidad sobre predicciones ya emitidas por MLP V7 y reentrenamiento independiente de modelos base para tres particiones.

## 1. Resumen ejecutivo

La estabilidad del índice compuesto depende del criterio de ordenamiento. Los rankings acumulados se mantienen en gran medida cuando se alteran los pesos; los rankings relativos cambian más cuando se eliminan los desenlaces graves. Se compararon Random Forest y HistGradientBoosting (más una referencia de media histórica) en particiones aleatoria, de grupos y temporal, pero **no se reentrenó la MLP** en estas particiones: no atribuir a la red neuronal las métricas temporales o por grupos.

## 2. Diseño del análisis de sensibilidad

Para cada uno de seis escenarios se aplicaron nuevos pesos a las seis predicciones de conteos normalizadas de V7, manteniendo invariables las estimaciones individuales. Se reprodujo la suma por departamento–municipio–actividad y el ranking relativo por total de trabajadores registrados con filtros de 2 registros y 10 trabajadores promedio. Para clasificar los cuatro niveles se recalcularon en cada escenario los percentiles P60, P90 y P97. No se volvió a entrenar ningún modelo.

Pesos en el orden [accidentes, muertes AT, invalidez AT, invalidez EL, incapacidad AT, incapacidad EL]:

- Original: [1, 5, 4, 4, 3, 3].
- Uniforme: [1, 1, 1, 1, 1, 1].
- Jerarquía suavizada: [1, 4, 3, 3, 2, 2].
- Muertes mayor peso: [1, 8, 4, 4, 3, 3].
- Solo accidentes: [1, 0, 0, 0, 0, 0].
- Sin muertes ni invalidez AT: [1, 0, 0, 4, 3, 3].

| Escenario | Top100 total compartido | Top100 relativo compartido | Cambio de nivel (%) | Alto/Crítico vs resto cambia (%) |
|---|---:|---:|---:|---:|
| Oficial V7 | 100/100 | 100/100 | 0.00 | 0.00 |
| Pesos iguales | 99/100 | 83/100 | 4.46 | 1.27 |
| Jerarquia suavizada | 99/100 | 98/100 | 0.53 | 0.14 |
| Muertes mas peso | 99/100 | 98/100 | 0.82 | 0.21 |
| Solo accidentes | 94/100 | 51/100 | 14.93 | 5.30 |
| Sin muertes e invalidez AT | 92/100 | 51/100 | 15.45 | 5.42 |

**Interpretación:** el ranking total conserva 92–99 integrantes del Top100 en los escenarios alternativos; el relativo conserva 51–98. El caso «solo accidentes» cambia el nivel de riesgo de 14,93 % de los registros. Esto es sensibilidad a la composición del índice, no validación empírica de la gravedad real de los seis desenlaces.

## 3. Diseño de validación adicional para accidentes

- **Aleatorio_80_20**: entrenamiento 132,250; prueba 33,063; periodos entrenamiento 2025-09,2025-11,2026-02; prueba 2025-09,2025-11,2026-02; grupos compartidos 25263; casos positivos en test 1846; media real test 0.16965.
- **Grupos_excluyentes_80_20**: entrenamiento 132,146; prueba 33,167; periodos entrenamiento 2025-09,2025-11,2026-02; prueba 2025-09,2025-11,2026-02; grupos compartidos 0; casos positivos en test 1893; media real test 0.14367.
- **Temporal_2025_a_feb2026**: entrenamiento 104,994; prueba 60,319; periodos entrenamiento 2025-09,2025-11; prueba 2026-02; grupos compartidos 56881; casos positivos en test 2927; media real test 0.13424.

Modelos: promedio histórico, RandomForestRegressor (80 árboles, profundidad máxima 18 y mínimo 5 observaciones por hoja) e HistGradientBoostingRegressor (pérdida Poisson, 120 iteraciones), entrenados de cero por partición. Se codificaron categorías solo con el conjunto de entrenamiento para evitar el acceso a categorías de prueba durante el ajuste. Se aplicó el tope de predicción calculado en el respectivo entrenamiento.

| Partición | Modelo | MAE | RMSE | R² | Desviación Poisson |
|---|---|---:|---:|---:|---:|
| Aleatorio_80_20 | Promedio_historico | 0.3102 | 2.4575 | -0.0000 | 1.3259 |
| Aleatorio_80_20 | Random_Forest | 0.1564 | 1.8560 | 0.4296 | 0.3173 |
| Aleatorio_80_20 | HistGradientBoosting_Poisson | 0.2448 | 2.0816 | 0.2825 | 0.5130 |
| Grupos_excluyentes_80_20 | Promedio_historico | 0.2896 | 1.3620 | -0.0002 | 1.0065 |
| Grupos_excluyentes_80_20 | Random_Forest | 0.1479 | 0.9555 | 0.5077 | 0.2910 |
| Grupos_excluyentes_80_20 | HistGradientBoosting_Poisson | 0.2429 | 1.3892 | -0.0406 | 0.4880 |
| Temporal_2025_a_feb2026 | Promedio_historico | 0.2928 | 1.9806 | -0.0004 | 1.0607 |
| Temporal_2025_a_feb2026 | Random_Forest | 0.1389 | 1.4956 | 0.4295 | 0.2580 |
| Temporal_2025_a_feb2026 | HistGradientBoosting_Poisson | 0.2276 | 1.7149 | 0.2499 | 0.4368 |

**Importante:** las métricas entre particiones no son directamente comparables como si fueran el mismo conjunto de prueba. Febrero de 2026 tiene prevalencia y media de accidentes inferiores a las del conjunto histórico de entrenamiento, y en la prueba por grupos los segmentos no se repiten entre entrenamiento y prueba. No generalizar la validación temporal a todo 2026 ni a periodos continuos.

## 4. Consecuencias para la tesis

1. La MLP V7 original mantiene únicamente las métricas de prueba aleatoria ya documentadas (MAE≈0,1570; R²≈0,4009). No corresponde declarar probada su capacidad prospectiva con este informe.
2. El análisis de sensibilidad de ponderaciones es adecuado como comprobación de robustez interna del índice, aunque la coincidencia de variables graves en dos periodos sigue afectando su interpretación sustantiva.
3. Para validar temporal y por grupos la MLP deben entrenarse de nuevo sus pesos en cada partición, sin información del conjunto de prueba y con codificación/estandarización ajustadas al entrenamiento. Se entrega script independiente con la opción `--mlp` para ejecutar localmente si se dispone de TensorFlow.
4. El esquema temporal se basa solo en septiembre y noviembre de 2025 como entrenamiento y febrero de 2026 como prueba; no existe una serie mensual consecutiva.
5. Los niveles percentilares son relativos, por lo que cambios en pesos y distribución de índice alteran los umbrales de cada escenario.
6. Los datos de muertes e invalidez de noviembre/febrero tienen coincidencias sistemáticas con accidentes y deben mantenerse como limitación explícita, incluso si se preservan por su procedencia publicada.

## 5. Metodología para reproducir y archivos

- `validacion_temporal_y_grupos_V7.py`: ejecutar con `python validacion_temporal_y_grupos_V7.py "ruta_al_csv.csv" --mlp` en el entorno con TensorFlow; sin `--mlp`, reproduce los baselines.
- `analisis_sensibilidad_V7_reproducible.py`: ejecución opcional con `python analisis_sensibilidad_V7_reproducible.py "10_predicciones_completas_herramienta_sst.xlsx"`; genera resultados de seis escenarios.
- `analisis_sensibilidad_V7/01_resumen_sensibilidad.csv`: resultados de seis escenarios.
- `analisis_sensibilidad_V7/02_top100_escenarios.csv`: detalle de los primeros 100 segmentos en cada ranking y escenario.
- `analisis_sensibilidad_V7/03_niveles_escenarios.csv`: clases por escenario.
- `validacion_adicional_V7/01_metricas_particiones.csv`: desempeño por modelo y partición.
- `validacion_adicional_V7/02_caracteristicas_particiones.csv`: composición de cada entrenamiento y prueba.
- `Informe_Auditoria_Resultados_V7_165313.md`: evaluación previa sobre ejecución MLP original.


**Nota final:** la presente evaluación no modifica la base, los datos observados de Positiva ni el código operativo de la Herramienta V7. Los scripts son pruebas complementarias reproducibles y sus resultados deben interpretarse de acuerdo con su alcance.
