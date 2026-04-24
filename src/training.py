"""
Training pipeline para el proyecto Boston BQ Predictor.

Entrena todos los modelos candidatos, los guarda como trained_model_n.pkl,
y guarda el modelo final elegido como final_model.pkl.

Uso:
    python src/training.py

Requiere:
    data/train/train_features.csv (generado por data_processing.py)

Genera:
    models/trained_model_1.pkl  (Logistic Regression)
    models/trained_model_2.pkl  (Decision Tree)
    models/trained_model_3.pkl  (Random Forest)
    models/trained_model_4.pkl  (XGBoost)
    models/final_model.pkl      (modelo final elegido)
    models/training_metrics.csv (métricas CV de cada modelo)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / 'data' / 'train'
MODELS_DIR = PROJECT_ROOT / 'models'

SCORING = {
    'f1_pos': 'f1',
    'precision_pos': 'precision',
    'recall_pos': 'recall',
    'pr_auc': 'average_precision',
    'roc_auc': 'roc_auc',
}


# ============================================================================
# FUNCIONES
# ============================================================================

def load_train_data():
    """Carga features de entrenamiento y separa X/y."""
    train = pd.read_csv(TRAIN_DIR / 'train_features.csv')
    X = train.drop(columns=['es_BQ'])
    y = train['es_BQ']
    if 'Year' in X.columns:
        X = X.drop(columns=['Year'])
    return X, y


def evaluate_with_cv(model, X, y, cv):
    """Ejecuta CV y devuelve métricas resumidas."""
    cv_out = cross_validate(model, X, y, cv=cv, scoring=SCORING,
                            n_jobs=-1, return_train_score=False)
    return {
        metric: (cv_out[f'test_{metric}'].mean(), cv_out[f'test_{metric}'].std())
        for metric in SCORING.keys()
    }


def build_models():
    """Define los modelos candidatos a entrenar."""
    return {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, random_state=RANDOM_STATE
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=20,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            tree_method='hist', eval_metric='logloss',
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def print_metrics(name, metrics):
    print(f"\n  [{name}]")
    print(f"    F1 (pos):   {metrics['f1_pos'][0]:.4f} ± {metrics['f1_pos'][1]:.4f}")
    print(f"    PR-AUC:     {metrics['pr_auc'][0]:.4f}")
    print(f"    ROC-AUC:    {metrics['roc_auc'][0]:.4f}")
    print(f"    Precision:  {metrics['precision_pos'][0]:.4f}")
    print(f"    Recall:     {metrics['recall_pos'][0]:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("TRAINING PIPELINE — Boston BQ Predictor")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1/4] Cargando datos de entrenamiento...")
    X, y = load_train_data()
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape} | % BQ: {y.mean()*100:.2f}%")

    # 2. Entrenar modelos
    print("\n[2/4] Entrenando modelos candidatos con CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    all_metrics = []

    for i, (name, model) in enumerate(models.items(), start=1):
        print(f"\n  Entrenando modelo {i}/{len(models)}: {name}...")
        metrics = evaluate_with_cv(model, X, y, cv)
        print_metrics(name, metrics)

        # Entrenar sobre todo el train y guardar
        model.fit(X, y)
        model_path = MODELS_DIR / f'trained_model_{i}.pkl'
        joblib.dump(model, model_path)

        # Acumular métricas para CSV
        row = {'model_id': i, 'model_name': name}
        for metric, (mean, std) in metrics.items():
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
        all_metrics.append(row)

    # 3. Elegir modelo final
    print("\n[3/4] Seleccionando modelo final...")
    metrics_df = pd.DataFrame(all_metrics)
    best_idx = metrics_df['f1_pos_mean'].idxmax()
    best_name = metrics_df.loc[best_idx, 'model_name']
    best_model_path = MODELS_DIR / f"trained_model_{metrics_df.loc[best_idx, 'model_id']}.pkl"
    final_model = joblib.load(best_model_path)

    # Guardar como final
    joblib.dump(final_model, MODELS_DIR / 'final_model.pkl')
    metrics_df.to_csv(MODELS_DIR / 'training_metrics.csv', index=False)

    print(f"  Modelo final elegido: {best_name}")
    print(f"  F1: {metrics_df.loc[best_idx, 'f1_pos_mean']:.4f}")

    # 4. Resumen
    print("\n[4/4] Guardando resultados...")
    print(f"  Modelos entrenados en: {MODELS_DIR}")
    for i, row in metrics_df.iterrows():
        print(f"    trained_model_{row['model_id']}.pkl ({row['model_name']})")
    print(f"    final_model.pkl ({best_name})")
    print(f"    training_metrics.csv")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETADO")
    print("=" * 60)


if __name__ == '__main__':
    main()