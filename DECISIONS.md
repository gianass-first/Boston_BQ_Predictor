# Decisiones de Diseño

Documento vivo que registra las decisiones tomadas durante el proyecto y su justificación. Cada decisión tiene fecha y contexto.

---

## Decisión 1 — Target: BQ nominal (no admisión real)

**Fecha:** 20-abr-2026

**Decisión:** El target `es_BQ` = 1 si `Finish ≤ Standard` para la franja edad/género del corredor, 0 en caso contrario.

**Alternativa considerada:** Usar "BQ real" aplicando un buffer de ~5 min al estándar nominal para reflejar el cutoff efectivo de admisión.

**Por qué NO el BQ real:**
- El dataset no etiqueta quiénes fueron realmente admitidos a Boston
- Aplicar un buffer sería una aproximación y añadiría una asunción arbitraria
- Alcance realista del capstone (2 semanas, 2h/día)

**Mitigación:**
- Documentar esta limitación upfront en la slide 2 de la presentación
- Mencionar que el modelo puede adaptarse fácilmente a "BQ-5min" cambiando una línea

---

## Decisión 2 — Features a incluir

**Fecha:** 20-abr-2026

### Features confirmadas

| Feature | Tipo | Justificación |
|---|---|---|
| `Age` | Numérica | Predictor obvio; el rendimiento depende fuertemente de la edad |
| `Gender` | Categórica (M/F) | Determinante del estándar BQ aplicable |
| `Race` | Categórica | Captura dificultad del circuito (desnivel, clima, pace culture) |
| `Country` | Categórica | Proxy de cultura de running, acceso a entrenamiento, etc. |

### Features NO incluidas

| Feature | Por qué no |
|---|---|
| `Finish` (tiempo final) | **Data leakage directo** — es lo que define el target |
| `Overall Place`, `Gender Place` | **Data leakage** — son función del tiempo final |
| `Standard` (tiempo BQ por categoría) | **Data leakage** — es lo que se compara para construir el target. Sin embargo, el modelo aprenderá las categorías implícitamente vía Age+Gender |
| `Year` | Usado para split temporal, no como feature (ver Decisión 3) |
| `Zip`, `City`, `State` (del corredor) | 94%, 62% y 38% de nulos respectivamente; baja calidad |
| `Name` | No predictivo y con implicaciones de privacidad |

### Features derivadas exploradas en notebook 03

- `Race_Category` (de Races.csv: Steep / Moderate / Minor) — captura dificultad
- `Race_Finishers` (de Races.csv) — tamaño del evento, proxy de prestigio
- `Age_Bracket` (derivada de Age) — alineada con categorías BQ
- `Is_Home_Country` — si el corredor corre en su país (boolean)

---

## Decisión 3 — Split temporal en lugar de random split

**Fecha:** 20-abr-2026

**Decisión:** Train = 2022-2023 / Test = 2024. No usamos `train_test_split` random.

**Justificación:**
- La tasa BQ creció año a año (11.3% → 13.0% → 13.9%) → hay drift temporal
- Un split temporal simula el uso real del modelo: entrenado con datos pasados, aplicado al futuro
- Evita que el modelo "haga trampa" al tener acceso a patrones específicos de 2024 durante el entrenamiento
- Validación cruzada dentro del train se hará con `TimeSeriesSplit` o k-fold estándar (el train ya es pasado, no hay leakage dentro)

**Nota:** Por eso `Year` no va como feature — sería darle pistas directas al modelo sobre a qué "era" pertenece cada registro.

---

## Decisión 4 — Muestreo estratificado de 300k filas

**Fecha:** 20-abr-2026

**Decisión:** Trabajar con una muestra estratificada de 300k filas (sobre las ~1.6M limpias), estratificando por `es_BQ` × `Year` × `Gender` para preservar las distribuciones clave.

**Alternativas consideradas:**

1. **Usar el dataset completo (1.6M):** Descartado por velocidad de iteración en portátil personal. XGBoost/RF sobre 1.6M tarda minutos, no segundos.
2. **Usar solo carreras españolas (~34k):** Descartado por:
   - Poca variedad (solo 4 carreras únicas)
   - Bias de selección (Sevilla 2024 tiene 31.6% BQ, 3x el promedio global)
   - Pérdida de generalización del modelo

**Justificación del tamaño (300k):**
- Suficiente para ensembles robustos y cross-validation
- Rápido de iterar (~segundos por entrenamiento)
- 300k × 12.76% baseline ≈ 38k positivos → suficiente para aprender la clase minoritaria

---

## Decisión 5 — España como slice narrativo, no como training set

**Fecha:** 20-abr-2026

**Decisión:** Las carreras celebradas en España (Madrid RnR, Barcelona, Sevilla) y los corredores con Country='ES' se reservan para análisis post-modelo (notebook 09) y la demo de Streamlit, no para entrenamiento específico.

**Justificación:**
- Conserva la generalidad del modelo
- Genera storytelling potente en la presentación: "¿qué maratón española te da más probabilidad de BQ?"
- Permite análisis de slice/fairness: ¿el modelo es igual de preciso en carreras españolas que en el dataset completo?

---

## Decisión 6 — Criterios de limpieza de datos

**Fecha:** 20-abr-2026

| Filtro aplicado | Razón | Impacto |
|---|---|---|
| `Finish > 0` | Excluir DNF (Did Not Finish) | -14.300 filas |
| `Gender ∈ {M, F}` | Solo géneros con estándar BQ definido | -24.499 filas (U, NB, X, NOT SPECIFIED) |
| `18 ≤ Age ≤ 85` | Rangos humanos razonables; BQ Standards van de Under 35 a 80+ | -106.974 filas |
| `Include = 'Yes'` en Races.csv | Respetar filtrado del autor original | Aplicado en notebook 01 |
| Normalización de `Country` | US/USA, GB/GBR, DE/GER aparecen como distintos | Unificar códigos ISO |

**No aplicado:**
- No se imputan valores faltantes en Country → se usa categoría "UNKNOWN"
- No se eliminan outliers de Finish (algunos tiempos > 10h son reales, aunque raros)

---

## Decisión 7 — Métricas de evaluación

**Fecha:** 20-abr-2026

**Decisión:** Las métricas principales serán **F1-score (clase positiva)** y **PR-AUC**.

**Por qué NO accuracy:**
- Con 87% de clase negativa, un modelo que predice "todos no-BQ" tendría 87% accuracy sin aprender nada

**Por qué F1 y PR-AUC:**
- Apropiadas para desbalance de clases (1:6.8)
- Priorizan el rendimiento en la clase minoritaria, que es la de interés
- PR-AUC es más informativa que ROC-AUC cuando la clase positiva es rara

**Métricas complementarias a reportar:**
- Precision y Recall (desglosadas)
- Matriz de confusión
- Análisis de calibración (¿las probabilidades predichas son fiables?)
- Slice metrics: rendimiento por género, por franja de edad, por país

---

## Decisión 8 — Manejo del desbalance

**Fecha:** 20-abr-2026

**Decisión:** Comparar sistemáticamente 3 enfoques en el notebook 06:

1. **Sin balanceo** (baseline)
2. **`class_weight='balanced'`** (ajuste de pesos)
3. **SMOTE** (oversampling sintético de la clase minoritaria)

El mejor según F1-CV se elige para el modelo final.

**No usar:**
- Undersampling fuerte → perderíamos demasiada información con solo 300k filas ya muestreadas

---

## Decisión 9 — Interpretación del drift temporal

**Fecha:** 21-abr-2026

El % BQ sube de 11.3% en 2022 a 13.9% en 2024 (+2.6 puntos porcentuales en 3 años).

**Hipótesis:** los estándares BQ nominales son estáticos (no se actualizan anualmente), pero la competitividad del corredor medio aumenta por factores externos:

- **Super-zapatillas de carbono** democratizadas desde ~2020 → todo el mundo corre 2-4% más rápido
- **Entrenamiento con datos** masificado (Garmin, Strava, COROS) → optimización sistemática del training
- **Selección post-pandemia** → corredores menos serios abandonaron, se quedaron los más comprometidos
- **Boom mediático del running** → atrae a deportistas con base física de otras disciplinas

**Implicación para el modelo:** las probabilidades predichas pueden infraestimar el % BQ real en predicciones futuras si no se reentrena anualmente. Esto refuerza la decisión del split temporal (Decisión 3) y la necesidad de reportar métricas en train-CV y test por separado.

---

## Decisión 10 — Feature Engineering (Notebook 03)

**Fecha:** 22-abr-2026

### Features eliminadas por leakage

- `Finish`: tiempo final, usado para construir el target
- `Standard`: umbral BQ por categoría, usado para construir el target
- `Overall Place`, `Gender Place`: función directa del tiempo final

### Features eliminadas por redundancia

- `Age Bracket`: discretización de Age
- `Race_City`, `Race_State`: redundantes con Race
- `Finishers`: redundante con Race_te (captura info similar)

### Encoding de Country

Agrupación de países con menos de 500 corredores como "Other". Reduce de ~222 a ~25-30 categorías útiles.

**Umbral decidido empíricamente:** corta ruido estadístico sin perder señal de los países principales. Aplicado con estadísticas de train y extendido a test (sin leakage).

### Encoding de Race (K-Fold Target Encoding con Smoothing)

`Race` tiene ~500 valores únicos. Opciones consideradas:

- **One-hot:** 500 columnas, ineficiente
- **Target encoding simple:** causaría leakage (usa el target en train)
- **K-Fold Target Encoding con Smoothing:** solución adoptada

**Implementación:**
- 5 folds para calcular encoding out-of-fold en train
- Smoothing bayesiano con factor 10 para carreras con pocas observaciones
- Para test, se usa el encoding calculado sobre TODO el train

**Fórmula del smoothing:** `encoding(race) = (n · mean_race + m · mean_global) / (n + m)` donde `n` = conteo de esa carrera, `m = 10` = factor de suavizado.

### Features derivadas añadidas

- **`Is_Home_Country`**: binaria. 1 si el corredor compite en su propio país. Proxy de ventaja local (menos jet lag, conocimiento del circuito, clima familiar).
- **`Race_Category_enc`**: ordinal (0=Minor, 1=Moderate, 2=Steep, 3=Very Steep). NaN imputado con la moda del train.
- **`Age_Squared`**: captura la no-linealidad del efecto edad. En el EDA vimos que los 60-69 años tienen mayor tasa de BQ que los 30-40. Los modelos lineales no pueden capturar esto solo con `Age`; los árboles lo capturan nativamente pero añadir la feature no hace daño.

### Artefactos guardados en `models/preprocessing_artifacts.joblib`

Para reutilización en predicciones futuras (Streamlit, inferencia):
- `countries_to_keep`, `race_country_map`, `category_mapping`
- `race_encoding_map` (encoding completo), `mode_category`, `global_mean_bq`, `feature_cols`

---

## Decisión 11 — Normalización exhaustiva de códigos de país

**Fecha:** 22-abr-2026

Al llegar al Notebook 04 se detectó que el one-hot encoding de `Country` contenía columnas duplicadas (BR/BRA, FR/FRA, ES/ESP, etc.) porque el `country_mapping` inicial del Notebook 01 (Decisión 6) solo cubría 3 países (USA, GBR, GER).

**Alcance de la corrección:**
- 166 códigos de 3 letras normalizados a ISO-2 (FRA→FR, BRA→BR, etc.)
- 5 códigos atípicos normalizados (NED→NL, SUI→CH, INA→ID, RSA→ZA, CHI→CL)
- 1 código ambiguo tratado como UNKNOWN (BUR: podría ser Burundi o Burkina Faso, solo 2 corredores)

**Resultado:** dataset pasa de ~220 códigos heterogéneos a 197 países en ISO-2 consistente (más UNKNOWN). El one-hot del Notebook 03 queda limpio sin duplicados.

**Aprendizaje:** la validación de cardinalidad de variables categóricas debe hacerse en el notebook de limpieza, no descubrirse en feature engineering. Añadir siempre un print de `value_counts()` y una verificación de longitud de strings como sanity check temprano.

## Decisión 12 — Manejo del desbalance: `scale_pos_weight` sobre SMOTE

**Fecha:** 27 de abril de 2026
**Notebook:** `06_imbalance_handling.ipynb`

### Contexto

Tras el N05 los modelos convergen a precision ~55% y recall ~10% con threshold 0.5. PR-AUC alto (~0.33) y ROC-AUC alto (~0.73) indican que el ranking interno del modelo es bueno. El problema no es de capacidad sino de **calibración + threshold**: las probabilidades están comprimidas hacia la prevalencia base (0.135), por lo que casi nadie cruza el umbral 0.5.

### Opciones evaluadas

Tres configuraciones de XGBoost con todo lo demás constante (mismo CV, mismos hiperparámetros, threshold fijo en 0.5):

1. **Baseline:** XGBoost sin manejo de desbalance.
2. **`scale_pos_weight = 6.4345`:** penalización en la pérdida (n_neg / n_pos).
3. **SMOTE clásico:** oversampling sintético dentro de `imblearn.pipeline.Pipeline` para evitar leakage.

### Resultados (CV 5-fold sobre train)

| Configuración | F1 | Precision | Recall | PR-AUC | ROC-AUC | Fit time |
|---|---|---|---|---|---|---|
| Baseline | 0.167 | 0.556 | 0.098 | 0.336 | 0.736 | 2.27s |
| scale_pos_weight | **0.349** | 0.235 | **0.677** | **0.336** | **0.735** | **1.84s** |
| SMOTE | 0.346 | 0.236 | 0.645 | 0.330 | 0.727 | 5.86s |

### Decisión: `scale_pos_weight`

Empate técnico en F1 con SMOTE (diferencia de 0.003, dentro del ruido), pero `scale_pos_weight` se elige por:

1. **Más rápido:** 3x menos tiempo de entrenamiento por fit.
2. **PR-AUC y ROC-AUC ligeramente superiores** (+0.007 y +0.008 vs SMOTE).
3. **Sin riesgo de generar perfiles imposibles:** SMOTE clásico interpola features categóricas one-hot (Country, Gender), produciendo combinaciones tipo "0.4 España + 0.6 Kenia" o "género 0.5", inexistentes en la realidad.
4. **Sin riesgo de leakage:** no hay datos sintéticos, simplifica el pipeline y la reproducibilidad.
5. **Implementación nativa de XGBoost:** un solo hiperparámetro con fórmula cerrada (`n_neg / n_pos`).

### Hallazgo clave

Las 3 configuraciones tienen **PR-AUC y ROC-AUC casi idénticos**. Esto significa que el manejo del desbalance NO mejora la capacidad de discriminación del modelo: solo desplaza las probabilidades hacia arriba para que más ejemplos crucen el threshold 0.5. La capacidad de ranking ya estaba en el baseline.

### Implicación

El siguiente cuello de botella ya no es el desbalance, es el **threshold de decisión**. Por eso el N07 hará threshold tuning sobre `XGBoost + scale_pos_weight` antes de cualquier nuevo intento de mejorar capacidad (GridSearch de hiperparámetros, ensembles, etc.).

### Descartado (no aplicado)

- **SMOTENC:** la solución correcta para features mixtas (numéricas + categóricas), pero requiere especificar manualmente las columnas categóricas y con el one-hot expandido es operativamente costoso. Dado que SMOTE clásico ya empata con `scale_pos_weight` y este último es preferible por velocidad y simplicidad, no se justifica el esfuerzo de SMOTENC.
- **Undersampling de la clase mayoritaria:** descartado porque desperdicia ~165k filas de información real de no-BQs.
- **ADASYN y otras variantes:** descartadas por el mismo motivo que SMOTE: no esperamos ganancia significativa sobre `scale_pos_weight` y aumentan complejidad.

## Decisión 13 — Threshold tuning: F0.5 sobre F1/F2

**Fecha:** 28 de abril de 2026  
**Notebook:** `07_threshold_tuning.ipynb`

### Contexto

Tras el N06, el modelo `XGBoost + scale_pos_weight` tenía precision 0.235 / recall 0.677 / F1 0.349 con threshold default 0.5. PR-AUC = 0.336 confirmaba buena capacidad de ranking, pero el threshold por defecto era arbitrario y sub-óptimo.

### Procedimiento

1. Probabilidades out-of-fold sobre todo el train con `cross_val_predict(method='predict_proba')`.
2. `precision_recall_curve` para todos los thresholds posibles.
3. Cálculo de F1, F2 y F0.5 para cada threshold.
4. Selección del threshold óptimo según criterio.

### Thresholds candidatos evaluados

| Criterio | Threshold | Precision | Recall | F-score correspondiente |
|---|---|---|---|---|
| Default | 0.500 | 0.235 | 0.677 | F1 = 0.349 |
| F1 óptimo | 0.590 | 0.298 | 0.470 | F1 = 0.365 |
| F2 óptimo (recall focus) | 0.420 | 0.206 | 0.799 | F2 = 0.507 |
| **F0.5 óptimo (precision focus)** | **0.748** | **0.465** | **0.214** | **F0.5 = 0.376** |

### Decisión: threshold = 0.748 (F0.5 óptimo)

Razones:

1. **Coherencia con el caso de uso de la app:** dar falsas esperanzas a un corredor (decir "vas a clasificar" cuando no) es peor que dar dudas razonables. F0.5 pondera precision el doble que recall, lo cual implementa esa lógica.
2. **Coherencia con el target del proyecto:** el target es BQ nominal (no admisión real a Boston). Un modelo "generoso" sobre un target ya generoso da un mensaje doblemente flojo. Threshold conservador compensa.
3. **Trade-off aceptado:** -46 pts de recall a cambio de +98% de precision. F1 sube ligeramente como bonus.

Los 3 thresholds (F1, F2, F0.5) se reportan en la presentación técnica como sensitivity analysis.

### Resultados sobre test 2024

| Métrica | Valor | Delta vs Baseline N05 |
|---|---|---|
| F1 | 0.330 | +0.154 (+87%) |
| Precision | 0.372 | -0.016 |
| Recall | 0.296 | +0.182 (+159%) |
| PR-AUC | 0.302 | — |
| ROC-AUC | 0.741 | — |

Drift temporal confirmado: precision en test cae 9 puntos vs CV (0.465 → 0.372) por la mayor proporción de corredores que se quedan a las puertas del BQ en 2024.

### Pendiente

Calibración de probabilidades (Platt / isotonic) para la app Streamlit, ya que `scale_pos_weight` deja las probabilidades sin calibrar (media en CV ≈ 0.43 vs prevalencia real 0.135). El threshold tuning no requiere calibración (solo ranking), pero la app sí.

---

## Decisión 14 — Limitación de generalización geográfica

**Fecha:** 28 de abril de 2026  
**Notebook:** `07_threshold_tuning.ipynb` (sección 8, slice por país)

### Hallazgo

Análisis de slice por país sobre test 2024 con threshold = 0.748:

| País | n | BQ rate | Precision | Recall | F1 |
|---|---|---|---|---|---|
| US | 54,093 | 13.7% | 0.372 | 0.405 | **0.388** |
| Other | 4,205 | 18.9% | 0.375 | 0.146 | 0.211 |
| CA | 3,250 | 11.9% | 0.550 | 0.028 | 0.054 |
| GB | 11,184 | 14.8% | 1.000 | 0.001 | 0.001 |

Sobre 11,184 corredores británicos (con ~1,650 BQs reales), el modelo predice 1 sólo BQ. Sobre 3,250 canadienses (con ~387 BQs), predice 3.

### Causa raíz

Sesgo de muestreo en los datos de entrenamiento: los maratones de `Races.csv` son mayoritariamente estadounidenses. Los británicos y canadienses representados son una minoría con perfiles muy específicos (probablemente runners de élite que viajan a USA), no representativos de la población general de runners de esos países. El modelo aprende que esos países "rara vez son BQ" y les asigna probabilidades sistemáticamente bajas.

### Implicación

**El modelo solo es fiable para corredores estadounidenses.** Para cualquier otra nacionalidad, las predicciones tienen un sesgo sistemático que no refleja la realidad sino la composición del dataset.

### Acción

1. Documentar limitación en la presentación técnica y en la memoria.
2. App Streamlit: limitar el caso de uso a corredores USA, o mostrar warning explícito para otras nacionalidades ("predicción menos fiable, dataset USA-céntrico").
3. Para futuras iteraciones: buscar datasets adicionales con maratones europeos (Berlín, Londres, Valencia) para reentrenar con datos más representativos.

## Decision 15 — Unsupervised Learning (Clustering of Runners)

**Date:** April 2026  
**Notebook:** `08_clustering_runners.ipynb`

### Objective

To identify **latent runner archetypes** using unsupervised learning in order to enrich the supervised prediction model with contextual segmentation.

The goal was not predictive performance, but **interpretability and user experience enhancement** in the Streamlit application.

---

### Methodology

We applied **KMeans clustering** using only:

- `Age`
- `Finish time`

### Critical design choice:

We explicitly excluded:

- `Country`
- `Gender`
- `Race category`

This ensures clusters represent **true performance archetypes**, not demographic or event bias.

---

### Why only 2 features?

- Avoid overfitting clustering to categorical variables
- Ensure clusters reflect physiological + performance structure
- Maximize interpretability for end users

---

### Scaling decision

StandardScaler was applied because:

- Age (18–80) and Finish (7200–25000 sec) are not comparable
- KMeans relies on Euclidean distance → scaling is mandatory

---

### Optimal number of clusters (k)

We evaluated k = 2 to 8 using:

- Elbow method (inertia)
- Silhouette score

### Final decision:

**k = 4 clusters**

| Criterion | Result |
|---|---|
| Silhouette score | 0.355 |
| Inertia improvement vs k=3 | Significant |
| Interpretability | High (4 clear archetypes) |

---

### Final clusters discovered

| Cluster | Profile | Interpretation |
|---|---|---|
| Advanced Young | Fast + young runners | Competitive amateurs |
| Advanced Veteran | Fast + experienced | Highest BQ success rate |
| Aspiring Young | Slow + young | Developing runners |
| Aspiring Veteran | Slow + older | Recreational runners |

---

### Key insight

> Marathon performance is a joint function of age and pacing efficiency, not a single linear skill dimension.

---

### Business value

In the Streamlit app:

- Each runner is assigned a cluster
- They receive:
  - “Runners like you have X% BQ rate”
  - “You are closer to cluster Y than Z”

This transforms the system into a **context-aware coaching tool**

---

## Decision 16 — Cluster Interpretation Strategy

**Date:** April 2026

### Approach

Clusters were not labeled algorithmically. Instead:

1. KMeans produces clusters
2. Interpretation based on:
   - Centroids
   - Age/Finish distributions
   - BQ rate per cluster
   - Demographic composition

---

### Why not automatic labeling?

Because clustering is **unsupervised** and semantic meaning must be validated post-hoc.

Manual interpretation ensures:

- Narrative consistency
- Business interpretability
- Avoid misleading labels

---

## Decision 17 — Separation of Clustering and Supervised Learning

**Date:** April 2026

### Decision

Clustering is NOT used as a feature in the predictive model.

---

### Reason

- Risk of indirect leakage (clusters derived from predictive variables)
- No guarantee clusters improve generalization
- Different objectives:
  - XGBoost → prediction
  - KMeans → interpretation

---

### Final architecture

1. **XGBoost model → probability of BQ**
2. **Threshold tuning → decision calibration**
3. **KMeans → contextual explanation layer**

---

## Decision 18 — Model Persistence Strategy

**Date:** April 2026

### Saved artifacts

- `kmeans_final.joblib`
- `scaler_clustering.joblib`
- `cluster_metadata.joblib`
- `finishers_with_clusters.parquet`

---

### Metadata includes:

- Cluster names
- Centroids (original scale)
- Feature list
- Silhouette score
- Dataset sizes

---

### Why this matters

Ensures:

- Reproducibility
- Production readiness (Streamlit)
- No retraining required for inference

---

## Decision 19 — Final System Architecture

**Date:** April 2026

### Final system design

| Layer | Function |
|---|---|
| XGBoost | Probability estimation |
| Threshold (0.748) | Decision boundary optimization |
| KMeans | Behavioral segmentation |

---

### Why this design works

Because the problem is not only predictive:

- Prediction alone is insufficient
- Decision threshold defines usefulness
- Clustering adds interpretability

---

### Key insight

> Real-world ML systems are not just models — they are decision pipelines.

---

## Decision 20 — Final Product Definition

**Date:** April 2026

### What this system is NOT

- Not a deterministic predictor
- Not a medical or physiological model
- Not a guarantee system

---

### What this system IS

> A probabilistic decision-support tool that estimates Boston Marathon qualification likelihood and contextualizes it within peer groups.

---

### Core value

- “What is my probability of BQ?”
- “What group of runners am I similar to?”
- “What would I need to improve to change cluster?”

---

## FINAL GLOBAL INSIGHT

Across all notebooks, the key conclusion is:

> The main limitation of the system is not model capacity, but decision calibration (thresholding) and data representativeness.

Everything else performs sufficiently well.
