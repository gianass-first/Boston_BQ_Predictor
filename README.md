# Boston Marathon BQ Predictor

**Proyecto Capstone de Machine Learning** · The Bridge Data Science & IA Bootcamp (Madrid, 2026)

**Autor:** Gian Marco Assandria

---

## El problema

¿Qué corredor alcanzará el tiempo clasificatorio (Boston Qualifier o "BQ") de la Maratón de Boston?

Este proyecto construye un modelo de clasificación binaria que, dado el perfil de un corredor (edad, género, país, carrera objetivo), predice la probabilidad de que alcance el tiempo BQ nominal correspondiente a su categoría edad/género.

**Ejemplo de uso:** un corredor de 32 años apuntando al Maratón de Valencia quiere saber, antes de entrenar para él, qué probabilidad tiene de clasificar a Boston según el perfil histórico de corredores similares en esa carrera.

## Dataset

Fuente: [Boston Marathon Qualifiers Dataset](https://www.kaggle.com/datasets/runningwithrock/boston-marathon-qualifiers-dataset) (Kaggle, Running with Rock, 2025).

- **1.76M registros** de resultados individuales de maratón
- **759 combinaciones Race × Year** únicas (tras limpieza)
- **3 años de datos**: 2022, 2023, 2024
- **3 tablas relacionadas:** Results, Races, BQStandards

Tras limpieza y muestreo estratificado:
- **Train:** 225.356 filas (2022-2023)
- **Test:** 74.644 filas (2024, reservado para evaluación final)
- **Tasa BQ base:** 13.45% (train) / 14.30% (test) — drift temporal confirmado
