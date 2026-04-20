# 🏃 Boston Marathon BQ Predictor

**Proyecto Capstone ML — The Bridge Data Science & IA Bootcamp (Madrid, 2026)**
**Autor:** Gian Marco

---

## 🎯 Problema

Clasificación binaria: dado el perfil de un corredor (edad, género, país, carrera objetivo), predecir la probabilidad de que alcance el tiempo Boston Qualifier (BQ) correspondiente a su categoría edad/género.

**Ejemplo de uso:** un corredor de 32 años apuntando al Maratón de Valencia quiere saber, antes de entrenar para él, qué probabilidad tiene de clasificar a Boston según el perfil histórico de corredores similares en esa carrera.

## 📊 Dataset

Fuente: [Boston Marathon Qualifiers Dataset](https://www.kaggle.com/datasets/runningwithrock/boston-marathon-qualifiers-dataset) (Kaggle, Rock Running, 2025).

- **1.76M registros** de resultados individuales de maratón
- **759 carreras únicas** (combinación Race × Year)
- **3 años**: 2022, 2023, 2024
- **3 tablas**: Results, Races, BQStandards

## 🏗️ Estructura del repo

```
boston-bq-predictor/
│
├── README.md                    # Este archivo
├── DECISIONS.md                 # Decisiones de diseño documentadas
├── requirements.txt             # Dependencias
├── .gitignore                   # Ignora data cruda, venv, cache
│
├── data/
│   ├── raw/                     # Datos originales de Kaggle (NO subir a Git)
│   │   ├── Results.csv
│   │   ├── Races.csv
│   │   └── BQStandards.csv
│   ├── processed/               # Datos limpios y procesados
│   │   ├── train.csv            # Split temporal 2022-2023
│   │   ├── test.csv             # Split temporal 2024
│   │   └── spain_slice.csv      # Slice para análisis narrativo
│   └── README.md                # Instrucciones para descargar datos
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb           # Limpieza + target + split
│   ├── 02_eda.ipynb                     # Exploración visual
│   ├── 03_feature_engineering.ipynb     # Features finales
│   ├── 04_baseline_models.ipynb         # Logistic Regression + Decision Tree
│   ├── 05_advanced_models.ipynb         # Random Forest + XGBoost/LightGBM
│   ├── 06_imbalance_handling.ipynb      # SMOTE vs class_weight
│   ├── 07_evaluation.ipynb              # Métricas + PR curves + análisis por subgrupos
│   ├── 08_interpretability.ipynb        # Feature importance + SHAP
│   └── 09_spain_slice_analysis.ipynb    # Análisis del hook narrativo
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Funciones de carga
│   ├── preprocessing.py         # Limpieza reutilizable
│   ├── features.py              # Feature engineering
│   ├── models.py                # Wrappers de modelos
│   └── evaluation.py            # Métricas custom
│
├── models/
│   └── best_model.pkl           # Modelo serializado final
│
├── app/
│   └── streamlit_app.py         # Demo interactiva
│
├── reports/
│   ├── figures/                 # Gráficos exportados
│   └── presentation.pdf         # Slides finales
│
└── tests/
    └── test_preprocessing.py    # Tests básicos (opcional pero suma)
```

## 🛠️ Stack técnico

- Python 3.12 (gestionado con UV)
- pandas, numpy
- scikit-learn
- XGBoost / LightGBM
- imbalanced-learn (SMOTE)
- seaborn, matplotlib
- SHAP (interpretabilidad)
- Streamlit (demo)

## 🚀 Cómo ejecutar

```bash
# Clonar el repo
git clone <repo-url>
cd boston-bq-predictor

# Crear entorno con UV
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Descargar datos (ver data/README.md)
# Correr notebooks en orden

# Lanzar demo
streamlit run app/streamlit_app.py
```

## 📈 Resultados clave

*(Se completa al final del proyecto)*

- Métrica objetivo: **F1-score clase positiva (BQ=1)** y **PR-AUC** (apropiadas para desbalance 1:6.8)
- Baseline (LogReg simple): _por definir_
- Mejor modelo: _por definir_

## ⚠️ Limitaciones conocidas

Ver [DECISIONS.md](./DECISIONS.md) para la justificación completa.

1. El target predice **cumplimiento del tiempo BQ nominal**, no admisión real al Boston Marathon (el cutoff real suele ser ≥5 min más estricto)
2. Dataset limitado a 2022-2024 (3 años post-pandemia)
3. No incluye datos de entrenamiento previos del corredor (ritmos, volumen, etc.)

## 📚 Agradecimientos

Dataset compilado y publicado por [Running with Rock](https://runningwithrock.com/).
