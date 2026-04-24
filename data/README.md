## Cómo obtener los datos

1. Descarga el dataset desde Kaggle:
   [Boston Marathon Qualifiers Dataset](https://www.kaggle.com/datasets/runningwithrock/boston-marathon-qualifiers-dataset)

2. Descomprime el archivo `archive.zip` y coloca los 3 CSVs en `data/raw/`:
   - `BQStandards.csv`
   - `Races.csv`
   - `Results.csv`

3. Ejecuta el notebook `notebooks/01_data_cleaning.ipynb` para generar los datos limpios en `data/processed/`:
   - `train.csv` (corredores 2022-2023 limpios)
   - `test.csv` (corredores 2024 limpios)
   - `spain_slice.csv` (slice narrativo para la presentación)
   - `cleaning_log.csv` (trazabilidad de la limpieza)

4. Ejecuta el notebook `notebooks/03_feature_engineering.ipynb` para generar las features finales en:
   - `data/train/train_features.csv` (listo para entrenar)
   - `data/test/test_features.csv` (reservado para evaluación final)

## Notas

- `data/processed/` contiene los datos limpios pre-feature-engineering
- `data/train/` y `data/test/` contienen las features finales listas para los modelos
- El test set NO se toca hasta el Notebook 07 (evaluación final)
EOF
