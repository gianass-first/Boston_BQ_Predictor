# Memoria del Proyecto — Boston BQ Predictor

**Autor:** Gian Marco Assandria
**Bootcamp:** Data Science & IA — TheBridge (Madrid)
**Módulo:** Machine Learning
**Fecha:** Mayo 2026
**Repositorio:** [github.com/gianass-first/Boston_BQ_Predictor](https://github.com/gianass-first)

---

## 1. Resumen ejecutivo

Boston BQ Predictor es un sistema predictivo que estima la probabilidad de que un corredor de maratón cualifique para el Boston Marathon (BQ, *Boston Qualifier*). El proyecto combina dos enfoques complementarios de Machine Learning:

- **Modelo supervisado (XGBoost):** clasificación binaria que predice si un runner alcanzará el corte BQ basándose en demografía y carrera elegida.
- **Modelo no supervisado (KMeans):** clustering que identifica cuatro arquetipos de runner para enriquecer la predicción con contexto narrativo.

El resultado se materializa en una aplicación Streamlit donde un runner introduce sus datos y recibe (a) su probabilidad individual de BQ, (b) su arquetipo de runner dentro de una población de más de un millón de finishers, y (c) una recomendación accionable de mejora.

---

## 2. Contexto y problema de negocio

### 2.1 ¿Qué es Boston Qualifier?

El Boston Marathon, organizado por la Boston Athletic Association (BAA), es una de las seis maratones World Marathon Majors y la única que exige tiempo de cualificación para inscribirse. Cada año, decenas de miles de runners compiten por una de las ~30.000 plazas, y el proceso de selección se basa en alcanzar tiempos mínimos (BQ standards) que varían por edad y género.

### 2.2 Standards BQ

La BAA publica una tabla de cortes BQ que se relajan progresivamente con la edad:

| Categoría | Hombres | Mujeres |
|-----------|---------|---------|
| Under 35 | 3h00m | 3h30m |
| 35-39 | 3h05m | 3h35m |
| 40-44 | 3h10m | 3h40m |
| ... | ... | ... |
| 80+ | 4h50m | 5h20m |

Esto significa que la edad no es una desventaja: un veterano puede cualificar con un tiempo significativamente peor que un joven.

### 2.3 Pregunta de negocio

¿Puedo predecir, antes de correr, si un runner tiene perspectivas reales de cualificar para Boston? ¿Y puedo darle contexto comparativo y recomendaciones útiles? Estas son las preguntas que el proyecto responde.

---

## 3. Datos

### 3.1 Origen

El dataset proviene de un repositorio público con resultados oficiales de maratones americanas y europeas entre 2015 y 2024. Tres ficheros principales:

- **`Results.csv`** (136 MB): un registro por corredor por carrera. Variables: `Year, Race, Name, Country, City, State, Gender, Age, Finish, Overall Place, Gender Place`.
- **`Races.csv`**: catálogo de carreras con su categoría de dificultad (Minor, Moderate, Steep, Very Steep) y flag de inclusión.
- **`BQStandards.csv`**: tabla oficial de cortes BQ por edad y género.

### 3.2 EDA principal

Hallazgos clave del análisis exploratorio (Notebook 02):

- **Tasa BQ global ~13.5%**, lo que implica un problema desbalanceado.
- **Sesgo geográfico fuerte**: ~70-80% del dataset son runners USA, lo que limita la generalización a otros mercados.
- **Drift temporal**: el rendimiento medio del runner amateur ha empeorado ligeramente entre 2015 y 2024 (la población maratoniana se ha popularizado, no solo eligen el reto los runners experimentados).
- **Distribución bimodal de tiempos**: dos picos identificables, uno en torno a 3h45 (runners competitivos) y otro en torno a 5h00 (runners populares).

### 3.3 Limitaciones del dataset

- Sesgo USA-céntrico (limitación principal).
- Solo runners con género binario M/F (BQ standards no contemplan género no binario).
- No incluye runners élite mundial (<2h30) en volumen suficiente para análisis.
- No incluye datos de entrenamiento previos del runner (kilometraje, ritmos, etc.), solo demografía y carrera.

---

## 4. Metodología

El proyecto se estructuró en 9 notebooks que reflejan un pipeline lineal de Machine Learning:

### 4.1 Notebook 01 — Data Cleaning

Filtrado por `Include == 'Yes'` (descarte de ultras y eventos atípicos), saneamiento de outliers en tiempo (rango 2h-7h) y edad (rango 12-90), eliminación de runners no finishers.

### 4.2 Notebook 02 — EDA

Análisis exploratorio con visualizaciones de distribución del target, drift temporal, sesgo geográfico, correlaciones y análisis por género/edad. Establecimiento de hipótesis de modelado.

### 4.3 Notebook 03 — Feature Engineering

Decisiones críticas:

- **Eliminación de variables leakage**: `Finish` y `Standard` se descartan del feature set del modelo supervisado porque se usaron para construir el target `es_BQ`.
- **Target encoding de `Race`**: cada carrera se codifica como su tasa BQ histórica (con K-fold para evitar overfitting).
- **One-hot encoding de `Country`**: 15 países con suficiente volumen + bucket "Other".
- **Variable `Age_Squared`** para capturar relación no lineal con el target.
- **Variable `Is_Home_Country`**: flag binario indicando si el runner corre en su país de origen.
- **Split temporal**: train (2015-2023) / test (2024) para simular condiciones de despliegue real.

### 4.4 Notebook 04 — Baseline Models

Entrenamiento de cinco modelos de referencia: Logistic Regression, Random Forest, XGBoost, LightGBM y KNN. Comparación de F1 sobre test 2024. **Resultado:** XGBoost lidera con margen, se selecciona para fases siguientes.

### 4.5 Notebook 05 — Advanced Models

Refinamiento del XGBoost con hiperparámetros más agresivos. Mejora marginal sobre el baseline.

### 4.6 Notebook 06 — Manejo del desbalance

Comparación de dos estrategias para abordar el desbalance ~13/87:

- **`class_weight='balanced'`** (intrínseco al algoritmo).
- **SMOTE** (oversampling sintético).

**Decisión:** `class_weight` como ganadora. Más simple, sin coste de generación sintética, resultados equivalentes a SMOTE en este problema.

### 4.7 Notebook 07 — Threshold tuning

El threshold por defecto (0.5) no es óptimo cuando los costes de los errores son asimétricos. Para una app que recomienda a un runner si tiene perspectivas BQ, **es preferible ser conservador**: más vale subestimar que dar falsas esperanzas.

Comparación de F1, F2 y F0.5 como criterios de optimización del threshold:

- **F2** prioriza recall (capturar más BQ reales).
- **F1** balance equilibrado.
- **F0.5** prioriza precision (estar seguros cuando decimos "BQ").

**Decisión:** F0.5, alineado con la lógica de producto. Threshold final: **0.748**.

Hallazgos secundarios documentados:

- **Drift temporal**: el modelo entrenado con datos antiguos pierde performance al predecir años recientes.
- **Sesgo geográfico**: predicciones menos fiables para runners no-USA.
- **Recomendación**: aplicar threshold reducido (0.5) para la franja 30-39 años, donde el modelo tiende a infravalorar la probabilidad real.

### 4.8 Notebook 08 — Clustering no supervisado

**Objetivo:** identificar arquetipos de runner para enriquecer la app con contexto narrativo, complementando la predicción individual del modelo supervisado.

**Configuración:**

- Universo: 1.046.358 finishers (post-saneamiento, género binario, edad realista).
- Muestra para fit: 150.000 estratificada por género y grupo de edad.
- Predict: aplicado al millón completo.
- Features: `Age` y `Finish` escaladas con StandardScaler.
- Algoritmo: KMeans con `k=4`.

**Justificación de k=4:**

- Silhouette = 0.355, casi idéntico al máximo en k=3 (0.359).
- Inertia 27% menor que k=3, captura estructura real adicional.
- Permite narrativa de cuatro arquetipos diferenciados, equilibrio entre granularidad y simplicidad para presentación.

**Los cuatro arquetipos:**

| Cluster | Edad media | Tiempo medio | % dataset | % BQ |
|---------|-----------|--------------|-----------|------|
| Joven Avanzado | 26 años | 3h44 | 28.7% | 19.0% |
| Veterano Avanzado | 46 años | 3h50 | 30.5% | **28.0%** |
| Joven Aspirante | 30 años | 5h22 | 23.0% | 0.0% |
| Veterano Aspirante | 53 años | 5h25 | 17.7% | 1.0% |

**Hallazgos centrales del clustering:**

1. **Los arquetipos cruzan edad × rendimiento**, no son una jerarquía única de nivel.
2. **La edad es palanca BQ tanto como la velocidad**: Veteranos Avanzados clasifican 9 puntos más que Jóvenes Avanzados a pesar de correr 6 minutos más despacio. Los cortes BQ se relajan con la edad.
3. **Muro vertical entre Aspirantes y Avanzados**: la tasa BQ pasa de 1% a 19%+. Cambiar de cluster es prerrequisito antes de pensar en BQ.
4. **Los Veteranos Avanzados son el cluster más internacional** (33% no-USA): runners experimentados que viajan a las grandes carreras americanas.

### 4.9 Notebook 09 — Conclusiones

Síntesis de hallazgos y validación cruzada de coherencia entre supervisado y clustering.

---

## 5. Modelo final y resultados

### 5.1 Métricas del modelo supervisado

| Modelo | Threshold | F1 (test 2024) | Precision | Recall |
|--------|-----------|----------------|-----------|--------|
| Baseline N05 (XGBoost sin tunear) | 0.5 | 0.176 | 0.388 | 0.114 |
| **Modelo final (XGBoost + class_weight + F0.5)** | **0.748** | **0.330** | 0.372 | 0.296 |
| Mejora relativa | — | **+87%** | -1.6 pts | **+18 pts** |

El modelo final mejora F1 en 87% sobre el baseline, principalmente vía un aumento sustancial de recall. La precisión se mantiene estable.

### 5.2 Validación del clustering

- **Silhouette score**: 0.355 sobre muestra de 30k runners.
- **Distribución de clusters**: ningún cluster es degenerado (mínimo 17.7%, máximo 30.5%).
- **Validación cualitativa**: los cuatro arquetipos son interpretables sociológicamente y consistentes con la cultura del running.

### 5.3 Lectura conjunta

El modelo supervisado y el clustering son complementarios, no competitivos:

- **Supervisado** da la **predicción precisa individual** ("32% probabilidad BQ").
- **Clustering** da el **contexto narrativo grupal** ("eres del cluster Veterano Avanzado, donde el 28% cualifica").

La combinación permite responder no solo "¿qué probabilidad tengo?" sino también "¿qué tipo de runner soy y qué runners parecidos a mí cualifican?".

---

## 6. Aplicación: Streamlit App

### 6.1 Funcionalidades

La app `streamlit_app.py` permite a un runner:

1. Introducir sus datos: edad, género, país, mejor tiempo de maratón, carrera donde corrió, año.
2. Recibir su **probabilidad de BQ** según el modelo supervisado, con threshold ajustado por edad.
3. Conocer su **arquetipo de runner** (cluster asignado por KMeans).
4. Visualizar su **posición** en el mapa de runners (scatter Age × Finish con los cuatro clusters).
5. Recibir una **sugerencia accionable** de mejora ("para pasar al cluster X, baja tu tiempo Y minutos").

### 6.2 Arquitectura

La app carga al arrancar:

- `final_model.pkl`: dict con el XGBoost final, threshold, feature names y métricas CV.
- `kmeans_final.joblib`: modelo de clustering.
- `scaler_clustering.joblib`: StandardScaler ajustado al muestreo del N08.
- `cluster_metadata.joblib`: nombres de clusters, centroides interpretables.
- `preprocessing_artifacts.joblib`: mappings de Race_te, países, categorías para reconstruir features al vuelo desde el input del usuario.
- `finishers_with_clusters.parquet`: dataset con clusters asignados, usado para calcular tasas BQ por cluster y dibujar el scatter.

### 6.3 Decisiones de UX

- **Una sola página con scroll**, no tabs: la narrativa "predicción → cluster → sugerencia" se cuenta mejor en lectura lineal.
- **Warning para runners no-USA**: se les avisa de que el modelo tiene sesgo geográfico.
- **Nota técnica explícita**: la predicción se basa en demografía y carrera, no en el tiempo previo del usuario (este se usa solo para asignar cluster).
- **Threshold dinámico por edad**: 0.5 para 30-39 años (donde el modelo tiende a infravalorar), 0.748 para el resto.

---

## 7. Limitaciones y trabajo futuro

### 7.1 Limitaciones reconocidas

1. **Sesgo USA-céntrico**: ~70-80% del dataset son runners USA. Las predicciones para otros países son menos fiables.
2. **Drift temporal**: la población maratoniana cambia con el tiempo. El modelo debería reentrenarse periódicamente con datos recientes.
3. **Género binario**: BQ standards no contemplan género no binario, por lo que el modelo no es inclusivo en este aspecto.
4. **Ausencia del cluster "élite mundial"**: runners sub-2h30 son <0.1% del dataset y no forman cluster propio.
5. **Sin datos de entrenamiento del runner**: el modelo no usa kilometraje semanal, ritmos en entrenamientos, historial de lesiones, etc.

### 7.2 Trabajo futuro

- **GridSearch sobre el modelo final** para optimización fina de hiperparámetros.
- **Ampliación del dataset** con maratones europeas y asiáticas para mitigar sesgo geográfico.
- **Modelo específico por franja de edad**: el rendimiento del modelo varía entre franjas, podría beneficiarse de modelos especializados.
- **Integración de datos de entrenamiento**: si el usuario aporta su kilometraje semanal o ritmos, podría mejorarse la precisión.
- **Predicción de "tiempo necesario para BQ"**: además de la probabilidad, calcular cuántos minutos tendría que mejorar para superar el threshold.

---

## 8. Conclusiones

El proyecto cumple los tres objetivos propuestos:

1. **Construye un modelo predictivo robusto** que mejora 87% sobre el baseline en F1, con threshold alineado a la lógica de producto.
2. **Aporta interpretabilidad narrativa** mediante clustering, identificando cuatro arquetipos de runner que enriquecen la predicción individual.
3. **Materializa el resultado** en una aplicación funcional que combina ambos enfoques en una experiencia de usuario clara y accionable.

Más allá del resultado técnico, el proyecto deja una lección metodológica importante: **clustering y supervisado no compiten, complementan**. El supervisado precisa, el clustering contextualiza. Juntos forman una respuesta más útil que cualquiera de los dos por separado.

El hallazgo más interesante a nivel de producto es la confirmación cuantitativa de que **la edad es palanca BQ tanto como la velocidad**. Un runner de 50 años con 3h50 tiene mejor perspectiva BQ que uno de 26 con 3h44, contraintuitivamente. Esta es la clase de insight que solo emerge de un análisis riguroso de datos masivos.

---

## 9. Anexos

### 9.1 Estructura del proyecto

```
Boston_BQ_Predictor/
├── app_streamlit/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   └── test/
├── models/
│   ├── final_model.pkl
│   ├── preprocessing_artifacts.joblib
│   └── clustering/
│       ├── kmeans_final.joblib
│       ├── scaler_clustering.joblib
│       └── cluster_metadata.joblib
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_class_weight_SMOTE.ipynb
│   ├── 07_threshold_tuning.ipynb
│   ├── 08_clustering_runners.ipynb
│   └── 09_final_conclusions.ipynb
├── reports/figures/
├── src/
├── tests/
├── DECISIONS.md
├── Memoria.md
├── README.md
└── requirements.txt
```

### 9.2 Stack técnico

- **Python 3.13** (gestor `uv`)
- **Pandas, NumPy** para manipulación de datos
- **Matplotlib, Seaborn** para visualización
- **Scikit-learn** para preprocessing, clustering y métricas
- **XGBoost** para el modelo supervisado final
- **Streamlit** para la aplicación
- **Jupyter** para los notebooks de desarrollo

### 9.3 Referencias

- Boston Athletic Association — Qualifying Standards: [www.baa.org](https://www.baa.org)
- Documentación del dataset original (Results.csv, Races.csv, BQStandards.csv)
- Notebooks del proyecto (01 al 09) para detalle metodológico completo
