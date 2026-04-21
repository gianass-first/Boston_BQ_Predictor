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

## Decisiones pendientes

- [ ] Threshold de decisión: ¿0.5 por defecto o optimizar según F1/precisión requerida?
- [ ] ¿Eliminar `Year` del set de features finales antes de entrenar? (está guardado por si acaso, pero no debería usarse como predictor)
