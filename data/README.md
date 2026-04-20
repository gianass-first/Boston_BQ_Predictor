# 📊 Datos del proyecto

Los archivos de datos NO están versionados en Git por su tamaño (Results.csv pesa 136 MB).

## Cómo obtener los datos

1. Descarga el dataset desde Kaggle:
   [Boston Marathon Qualifiers Dataset](https://www.kaggle.com/datasets/runningwithrock/boston-marathon-qualifiers-dataset)

2. Descomprime el archivo `archive.zip` y coloca los 3 CSVs en `data/raw/`:
   - `BQStandards.csv`
   - `Races.csv`
   - `Results.csv`

3. Ejecuta el notebook `notebooks/01_data_cleaning.ipynb` para generar los datos procesados en `data/processed/`:
   - `train.csv`
   - `test.csv`
   - `spain_slice.csv`
   - `cleaning_log.csv`
