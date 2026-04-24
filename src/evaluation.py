"""
Evaluation pipeline para el proyecto Boston BQ Predictor.

Carga el modelo final entrenado y lo evalúa sobre el conjunto de test.
Genera métricas de evaluación y matriz de confusión.

Uso:
    python src/evaluation.py

Requiere:
    data/test/test_features.csv (generado por data_processing.py)
    models/final_model.pkl (generado por training.py)

Genera:
    models/evaluation_report.csv (métricas sobre test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / 'data' / 'test'
MODELS_DIR = PROJECT_ROOT / 'models'


# ============================================================================
# FUNCIONES
# ============================================================================

def load_test_data():
    """Carga features de test y separa X/y."""
    test = pd.read_csv(TEST_DIR / 'test_features.csv')
    X = test.drop(columns=['es_BQ'])
    y = test['es_BQ']
    if 'Year' in X.columns:
        X = X.drop(columns=['Year'])
    return X, y


def compute_metrics(y_true, y_pred, y_proba):
    """Calcula todas las métricas principales."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_pos': f1_score(y_true, y_pred),
        'precision_pos': precision_score(y_true, y_pred),
        'recall_pos': recall_score(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_proba),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }


def print_confusion_matrix(y_true, y_pred):
    """Imprime la matriz de confusión con etiquetas interpretables."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    print("\n  Matriz de confusión:")
    print(f"                  Predicho=No-BQ    Predicho=BQ")
    print(f"    Real=No-BQ         {tn:>7,}        {fp:>7,}")
    print(f"    Real=BQ            {fn:>7,}        {tp:>7,}")
    print(f"\n    Total predicciones: {total:,}")
    print(f"    Verdaderos positivos (BQ detectados): {tp:,}")
    print(f"    Falsos negativos (BQ que se perdieron): {fn:,}")
    print(f"    Falsos positivos (no-BQ etiquetados como BQ): {fp:,}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("EVALUATION PIPELINE — Boston BQ Predictor")
    print("=" * 60)

    # 1. Cargar datos de test
    print("\n[1/4] Cargando datos de test...")
    X_test, y_test = load_test_data()
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape} | % BQ: {y_test.mean()*100:.2f}%")

    # 2. Cargar modelo final
    print("\n[2/4] Cargando modelo final...")
    model_path = MODELS_DIR / 'final_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Ejecuta primero 'python src/training.py'."
        )
    model = joblib.load(model_path)
    print(f"  Cargado: {type(model).__name__}")

    # 3. Predicciones y métricas
    print("\n[3/4] Generando predicciones y calculando métricas...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("\n  Métricas sobre TEST SET (datos de 2024, no vistos en entrenamiento):")
    print(f"    Accuracy:     {metrics['accuracy']:.4f}")
    print(f"    F1 (pos):     {metrics['f1_pos']:.4f}")
    print(f"    Precision:    {metrics['precision_pos']:.4f}")
    print(f"    Recall:       {metrics['recall_pos']:.4f}")
    print(f"    PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:      {metrics['roc_auc']:.4f}")

    print_confusion_matrix(y_test, y_pred)

    print("\n  Classification report completo:")
    print(classification_report(y_test, y_pred,
                                 target_names=['No-BQ', 'BQ'],
                                 digits=4))

    # 4. Guardar reporte
    print("\n[4/4] Guardando reporte de evaluación...")
    report_df = pd.DataFrame([metrics])
    report_df.to_csv(MODELS_DIR / 'evaluation_report.csv', index=False)
    print(f"  Guardado: {MODELS_DIR / 'evaluation_report.csv'}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETADO")
    print("=" * 60)


if __name__ == '__main__':
    main()