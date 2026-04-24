"""
Data processing pipeline para el proyecto Boston BQ Predictor.

Ejecuta el pipeline completo:
1. Carga datos raw desde data/raw/
2. Aplica limpieza y normalización de países
3. Construye el target es_BQ
4. Genera split temporal train (2022-2023) / test (2024)
5. Aplica feature engineering (encoding, features derivadas, K-Fold target encoding)
6. Guarda resultados en data/train/ y data/test/

Uso:
    python src/data_processing.py

Requiere:
    data/raw/BQStandards.csv
    data/raw/Races.csv
    data/raw/Results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

RANDOM_STATE = 42
TARGET_SAMPLE_SIZE = 300_000
MIN_COUNTRY_COUNT = 500
SMOOTHING_FACTOR = 10
N_SPLITS_TE = 5

# Mapping exhaustivo de códigos de país (Decisión 11)
COUNTRY_MAPPING = {
    # ISO 3-letra estándar (comunes)
    'USA': 'US', 'GBR': 'GB', 'CAN': 'CA', 'FRA': 'FR', 'AUS': 'AU',
    'GER': 'DE', 'ITA': 'IT', 'IRL': 'IE', 'MEX': 'MX', 'RUS': 'RU',
    'ESP': 'ES', 'JPN': 'JP', 'ARG': 'AR', 'TWN': 'TW', 'DEN': 'DK',
    'SWE': 'SE', 'AUT': 'AT', 'BRA': 'BR', 'SIN': 'SG', 'BEL': 'BE',
    'HKG': 'HK', 'POL': 'PL', 'NZL': 'NZ', 'NOR': 'NO', 'SVK': 'SK',
    'POR': 'PT', 'CZE': 'CZ', 'IND': 'IN', 'EST': 'EE', 'FIN': 'FI',
    'CHN': 'CN', 'MAR': 'MA', 'TUR': 'TR', 'ISR': 'IL', 'PHI': 'PH',
    'THA': 'TH', 'COL': 'CO', 'ROU': 'RO', 'HUN': 'HU', 'KEN': 'KE',
    'LTU': 'LT', 'ISL': 'IS', 'UKR': 'UA', 'VIE': 'VN', 'KOR': 'KR',
    'PER': 'PE', 'VEN': 'VE', 'PAN': 'PA', 'TAN': 'TZ', 'ECU': 'EC',
    'DOM': 'DO', 'CRO': 'HR', 'ETH': 'ET', 'LUX': 'LU', 'EGY': 'EG',
    'UGA': 'UG', 'SRB': 'RS', 'BOL': 'BO',
    # ISO 3-letra menos comunes
    'MGL': 'MN', 'ERI': 'ER', 'GRE': 'GR', 'JOR': 'JO', 'CRC': 'CR',
    'IRI': 'IR', 'TRI': 'TT', 'KAZ': 'KZ', 'PUR': 'PR', 'LIE': 'LI',
    'ALG': 'DZ', 'LIB': 'LB', 'PAK': 'PK', 'BAN': 'BD', 'ZIM': 'ZW',
    'GEO': 'GE', 'GHA': 'GH', 'ANG': 'AO', 'AZE': 'AZ', 'NEP': 'NP',
    'BAH': 'BS', 'NCA': 'NI', 'TPE': 'TW', 'ARM': 'AM', 'KSA': 'SA',
    'KGZ': 'KG', 'LBA': 'LY', 'MLT': 'MT', 'MDA': 'MD', 'MON': 'MC',
    'SEN': 'SN', 'SRI': 'LK', 'TOG': 'TG', 'TRN': 'TT', 'TUN': 'TN',
    'BDI': 'BI', 'COD': 'CD', 'MAC': 'MO', 'CIV': 'CI', 'CAF': 'CF',
    'VIN': 'VC', 'MOZ': 'MZ', 'COM': 'KM', 'DMA': 'DM', 'DJI': 'DJ',
    'LAT': 'LV', 'BRN': 'BH', 'MAS': 'MY', 'BUL': 'BG', 'FRO': 'FO',
    'PAR': 'PY', 'QAT': 'QA', 'SWZ': 'SZ', 'GUA': 'GT', 'KOS': 'XK',
    'ESA': 'SV', 'SLO': 'SI', 'BLR': 'BY', 'CYP': 'CY', 'BIH': 'BA',
    'MDV': 'MV', 'URU': 'UY', 'HON': 'HN', 'GIB': 'GI', 'MKD': 'MK',
    'ZAM': 'ZM', 'GRL': 'GL', 'SUD': 'SD', 'CHA': 'TD', 'BRU': 'BN',
    'AND': 'AD', 'IRQ': 'IQ', 'ALB': 'AL', 'MRI': 'MU', 'PLE': 'PS',
    'BER': 'BM', 'GUM': 'GU', 'NGR': 'NG', 'JAM': 'JM', 'CUB': 'CU',
    'GRN': 'GD', 'AFG': 'AF', 'HAI': 'HT', 'SUR': 'SR', 'LCA': 'LC',
    'NAM': 'NA', 'UZB': 'UZ', 'SYR': 'SY', 'MNE': 'ME', 'CMR': 'CM',
    'OMN': 'OM', 'TKM': 'TM', 'BEN': 'BJ', 'SMR': 'SM', 'PNG': 'PG',
    'KUW': 'KW', 'CPV': 'CV', 'BIZ': 'BZ', 'MAD': 'MG', 'GUY': 'GY',
    'UAE': 'AE', 'CAM': 'KH', 'BOT': 'BW', 'LAO': 'LA', 'PLW': 'PW',
    'RWA': 'RW', 'MAW': 'MW',
    # Códigos atípicos
    'NED': 'NL', 'SUI': 'CH', 'INA': 'ID', 'RSA': 'ZA', 'CHI': 'CL',
    'BUR': 'UNKNOWN',
}

# Mapping de carreras a su país de celebración
RACE_COUNTRY_MAP = {
    'London Marathon': 'GB', 'Berlin Marathon': 'DE', 'Tokyo Marathon': 'JP',
    'RnR Madrid Marathon': 'ES', 'Zurich Marato Barcelona': 'ES',
    'Zurich Maraton Sevilla': 'ES', 'Zurich Marathon': 'CH',
    'Mexico City Marathon': 'MX', 'Marathon Pour Tous': 'FR',
    'Athens Marathon': 'GR', 'Amsterdam Marathon': 'NL',
    'Rotterdam Marathon': 'NL', 'Valencia Marathon': 'ES',
    'Paris Marathon': 'FR', 'Copenhagen Marathon': 'DK',
    'Dublin Marathon': 'IE', 'Stockholm Marathon': 'SE',
    'Vienna City Marathon': 'AT', 'Cape Town Marathon': 'ZA',
    'Comrades Marathon': 'ZA', 'Sydney Marathon': 'AU',
    'Melbourne Marathon': 'AU', 'Toronto Waterfront Marathon': 'CA',
    'Ottawa Marathon': 'CA', 'LANZAROTE INTERNATIONAL MARATHON': 'ES',
}

CATEGORY_MAPPING = {'Minor': 0, 'Moderate': 1, 'Steep': 2, 'Very Steep': 3}

# Rutas (relativas a la raíz del proyecto)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
TRAIN_DIR = PROJECT_ROOT / 'data' / 'train'
TEST_DIR = PROJECT_ROOT / 'data' / 'test'
MODELS_DIR = PROJECT_ROOT / 'models'


# ============================================================================
# FUNCIONES DE LIMPIEZA
# ============================================================================

def assign_age_bracket(age):
    """Asigna la franja de edad según categorías oficiales BQ."""
    if age < 35: return 'Under 35'
    elif age < 40: return '35-39'
    elif age < 45: return '40-44'
    elif age < 50: return '45-49'
    elif age < 55: return '50-54'
    elif age < 60: return '55-59'
    elif age < 65: return '60-64'
    elif age < 70: return '65-69'
    elif age < 75: return '70-74'
    elif age < 80: return '75-79'
    else: return '80 and Over'


def clean_results(results_df, bq_df, races_df):
    """Aplica limpieza y construye target es_BQ."""
    df = results_df.copy()

    # Age a numérico
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Filtros de limpieza
    df = df[df['Finish'] > 0]
    df = df[df['Gender'].isin(['M', 'F'])]
    df = df[(df['Age'] >= 18) & (df['Age'] <= 85)]

    # Drop columnas irrelevantes
    df = df.drop(columns=['Zip', 'City', 'State', 'Name'], errors='ignore')

    # Normalizar Country
    df['Country'] = df['Country'].replace(COUNTRY_MAPPING)
    df['Country'] = df['Country'].fillna('UNKNOWN')

    # Snapshot antes de filtrar por Include (para spain_slice)
    df_all_races = df.copy()
    df_all_races['Age Bracket'] = df_all_races['Age'].apply(assign_age_bracket)
    df_all_races = df_all_races.merge(bq_df, on=['Gender', 'Age Bracket'], how='left')
    df_all_races['es_BQ'] = (df_all_races['Finish'] <= df_all_races['Standard']).astype(int)

    # Merge con Races.csv (solo Include=Yes)
    races_clean = races_df[races_df['Include'] == 'Yes'].copy()
    races_clean = races_clean[['Year', 'Race', 'City', 'State', 'Finishers', 'Category']]
    races_clean = races_clean.rename(columns={'City': 'Race_City', 'State': 'Race_State'})
    df = df.merge(races_clean, on=['Year', 'Race'], how='inner')

    # Construcción del target
    df['Age Bracket'] = df['Age'].apply(assign_age_bracket)
    df = df.merge(bq_df, on=['Gender', 'Age Bracket'], how='left')
    df['es_BQ'] = (df['Finish'] <= df['Standard']).astype(int)

    return df, df_all_races


def build_spain_slice(df_all_races):
    """Construye el slice narrativo de carreras españolas."""
    spain_keywords = ['Madrid', 'Barcelona', 'Sevilla', 'Valencia', 'Lanzarote']
    mask = df_all_races['Race'].str.contains('|'.join(spain_keywords), case=False, na=False)
    mask |= df_all_races['Race'].str.contains('Zurich Marato', case=False, na=False)
    return df_all_races[mask].reset_index(drop=True)


def stratified_sample(df, target_size, random_state):
    """Muestreo estratificado por es_BQ × Year × Gender."""
    frac = target_size / len(df)
    strata = df['es_BQ'].astype(str) + '_' + df['Year'].astype(str) + '_' + df['Gender']
    return (
        df.groupby(strata, group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=random_state))
          .reset_index(drop=True)
    )


# ============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# ============================================================================

def kfold_target_encode(train_df, test_df, col, target, n_splits, smoothing, random_state):
    """Target encoding con K-Fold para evitar leakage."""
    global_mean = train_df[target].mean()
    train_encoded = np.zeros(len(train_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        agg = fold_train.groupby(col)[target].agg(['mean', 'count'])
        smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        val_cats = train_df.iloc[val_idx][col]
        train_encoded[val_idx] = val_cats.map(smoothed).fillna(global_mean).values

    agg_full = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed_full = (agg_full['count'] * agg_full['mean'] + smoothing * global_mean) / (agg_full['count'] + smoothing)
    test_encoded = test_df[col].map(smoothed_full).fillna(global_mean).values

    return train_encoded, test_encoded, smoothed_full


def get_race_country(race_name):
    """Asigna el país donde se celebra la carrera. Default: US."""
    return RACE_COUNTRY_MAP.get(race_name, 'US')


def engineer_features(train, test):
    """Aplica feature engineering: encoding, features derivadas, K-Fold TE."""

    # Drop columnas con leakage o redundantes
    cols_to_drop = ['Finish', 'Standard', 'Overall Place', 'Gender Place',
                    'Age Bracket', 'Race_City', 'Race_State']
    train = train.drop(columns=cols_to_drop, errors='ignore')
    test = test.drop(columns=cols_to_drop, errors='ignore')

    # Agrupar países de baja frecuencia
    country_counts = train['Country'].value_counts()
    countries_to_keep = country_counts[country_counts >= MIN_COUNTRY_COUNT].index.tolist()
    train['Country_grouped'] = train['Country'].where(train['Country'].isin(countries_to_keep), 'Other')
    test['Country_grouped'] = test['Country'].where(test['Country'].isin(countries_to_keep), 'Other')

    # Is_Home_Country
    train['Race_Country'] = train['Race'].apply(get_race_country)
    test['Race_Country'] = test['Race'].apply(get_race_country)
    train['Is_Home_Country'] = (train['Country'] == train['Race_Country']).astype(int)
    test['Is_Home_Country'] = (test['Country'] == test['Race_Country']).astype(int)

    # Race Category encoding
    train['Race_Category_enc'] = train['Category'].map(CATEGORY_MAPPING)
    test['Race_Category_enc'] = test['Category'].map(CATEGORY_MAPPING)
    mode_category = train['Race_Category_enc'].mode()[0]
    train['Race_Category_enc'] = train['Race_Category_enc'].fillna(mode_category)
    test['Race_Category_enc'] = test['Race_Category_enc'].fillna(mode_category)

    # K-Fold Target Encoding para Race
    race_enc_train, race_enc_test, race_encoding_map = kfold_target_encode(
        train, test, col='Race', target='es_BQ',
        n_splits=N_SPLITS_TE, smoothing=SMOOTHING_FACTOR, random_state=RANDOM_STATE
    )
    train['Race_te'] = race_enc_train
    test['Race_te'] = race_enc_test

    # Age_Squared
    train['Age_Squared'] = train['Age'] ** 2
    test['Age_Squared'] = test['Age'] ** 2

    # Gender binario
    train['Gender_M'] = (train['Gender'] == 'M').astype(int)
    test['Gender_M'] = (test['Gender'] == 'M').astype(int)

    # One-hot Country
    train_dummies = pd.get_dummies(train['Country_grouped'], prefix='Country', dtype=int)
    test_dummies = pd.get_dummies(test['Country_grouped'], prefix='Country', dtype=int)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    train = pd.concat([train, train_dummies], axis=1)
    test = pd.concat([test, test_dummies], axis=1)

    # Drop columnas originales ya encodeadas
    cols_encoded_drop = ['Race', 'Country', 'Country_grouped', 'Gender', 'Category',
                         'Race_Country', 'Finishers']
    train = train.drop(columns=cols_encoded_drop, errors='ignore')
    test = test.drop(columns=cols_encoded_drop, errors='ignore')

    # Reordenar target al final
    target_col = 'es_BQ'
    feature_cols = [c for c in train.columns if c != target_col]
    train = train[feature_cols + [target_col]]
    test = test[feature_cols + [target_col]]

    artifacts = {
        'countries_to_keep': countries_to_keep,
        'race_country_map': RACE_COUNTRY_MAP,
        'category_mapping': CATEGORY_MAPPING,
        'race_encoding_map': race_encoding_map,
        'mode_category': mode_category,
        'global_mean_bq': train['es_BQ'].mean(),
        'feature_cols': feature_cols,
    }

    return train, test, artifacts


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("DATA PROCESSING PIPELINE — Boston BQ Predictor")
    print("=" * 60)

    # 1. Cargar datos raw
    print("\n[1/5] Cargando datos raw...")
    results = pd.read_csv(RAW_DIR / 'Results.csv', low_memory=False)
    races = pd.read_csv(RAW_DIR / 'Races.csv')
    bq = pd.read_csv(RAW_DIR / 'BQStandards.csv')
    print(f"  Results: {len(results):,} filas")

    # 2. Limpieza + target
    print("\n[2/5] Limpieza y construcción del target...")
    df, df_all_races = clean_results(results, bq, races)
    print(f"  Tras limpieza: {len(df):,} filas")
    print(f"  % BQ: {df['es_BQ'].mean()*100:.2f}%")

    # 3. Muestreo estratificado
    print("\n[3/5] Muestreo estratificado...")
    df_sample = stratified_sample(df, TARGET_SAMPLE_SIZE, RANDOM_STATE)
    train = df_sample[df_sample['Year'].isin([2022, 2023])].reset_index(drop=True)
    test = df_sample[df_sample['Year'] == 2024].reset_index(drop=True)
    spain_slice = build_spain_slice(df_all_races)
    print(f"  Train: {len(train):,} | Test: {len(test):,} | Spain slice: {len(spain_slice):,}")

    # Guardar datos limpios
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_DIR / 'train.csv', index=False)
    test.to_csv(TEST_DIR / 'test.csv', index=False)
    spain_slice.to_csv(TEST_DIR / 'spain_slice.csv', index=False)

    # 4. Feature engineering
    print("\n[4/5] Feature engineering...")
    train_feat, test_feat, artifacts = engineer_features(train, test)
    print(f"  Features generadas: {len(artifacts['feature_cols'])}")

    # 5. Guardar resultados
    print("\n[5/5] Guardando resultados...")
    train_feat.to_csv(TRAIN_DIR / 'train_features.csv', index=False)
    test_feat.to_csv(TEST_DIR / 'test_features.csv', index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, MODELS_DIR / 'preprocessing_artifacts.joblib')

    print(f"\n  Guardado en data/train/: train.csv, train_features.csv")
    print(f"  Guardado en data/test/: test.csv, test_features.csv, spain_slice.csv")
    print(f"  Artefactos en models/preprocessing_artifacts.joblib")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)


if __name__ == '__main__':
    main()