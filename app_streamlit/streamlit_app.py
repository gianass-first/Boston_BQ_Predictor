"""
Boston BQ Predictor — App Streamlit

Combina:
- Modelo supervisado XGBoost (N07): probabilidad individual de BQ.
- Modelo de clustering KMeans (N08): arquetipo de runner.
- Preprocessing artifacts del N03: feature engineering exacto.
- Verificación BQ directa contra el corte oficial BAA.

Ejecutar desde la raíz del proyecto:
    streamlit run app_streamlit/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

st.set_page_config(
    page_title='Boston BQ Predictor',
    layout='centered',
)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
CLUSTERING_DIR = MODELS_DIR / 'clustering'
DATA_DIR = BASE_DIR / 'data'

THRESHOLD_DEFAULT = 0.748
THRESHOLD_30_39 = 0.5

# =====================================================================
# CARGA DE ARTEFACTOS
# =====================================================================

@st.cache_resource
def load_artifacts():
    artifacts = {}
    final = joblib.load(MODELS_DIR / 'final_model.pkl')
    artifacts['model'] = final['model']
    artifacts['model_threshold'] = final['threshold']
    artifacts['model_features'] = final['feature_names']
    artifacts['kmeans'] = joblib.load(CLUSTERING_DIR / 'kmeans_final.joblib')
    artifacts['scaler'] = joblib.load(CLUSTERING_DIR / 'scaler_clustering.joblib')
    artifacts['cluster_metadata'] = joblib.load(CLUSTERING_DIR / 'cluster_metadata.joblib')
    artifacts['preprocess'] = joblib.load(MODELS_DIR / 'preprocessing_artifacts.joblib')
    artifacts['df_clusters'] = pd.read_parquet(
        DATA_DIR / 'processed' / 'finishers_with_clusters.parquet'
    )

    # Traducir cluster_name de inglés a español
    name_mapping = {
        'Advanced Young':   'Joven Avanzado',
        'Advanced Veteran': 'Veterano Avanzado',
        'Aspiring Young':   'Joven Aspirante',
        'Aspiring Veteran': 'Veterano Aspirante',
    }
    artifacts['df_clusters']['cluster_name'] = (
        artifacts['df_clusters']['cluster_name'].astype(str).replace(name_mapping)
    )

    return artifacts


# =====================================================================
# UTILIDADES
# =====================================================================

def fmt_seconds(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f'{h}h{m:02d}m{sec:02d}s'


def time_str_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def get_bq_standard(age, gender):
    """Devuelve el corte BQ oficial en segundos según edad y género."""
    bq_table_m = [
        (35, 10800), (40, 11100), (45, 11400), (50, 12000), (55, 12300),
        (60, 12900), (65, 13800), (70, 14700), (75, 15600), (80, 16500),
        (200, 17400),
    ]
    bq_table_f = [
        (35, 12600), (40, 12900), (45, 13200), (50, 13800), (55, 14100),
        (60, 14700), (65, 15600), (70, 16500), (75, 17400), (80, 18300),
        (200, 19200),
    ]
    table = bq_table_m if gender == 'M' else bq_table_f
    for age_limit, standard in table:
        if age < age_limit:
            return standard
    return table[-1][1]


def predict_cluster(age, finish_seconds, scaler, kmeans, metadata):
    X = np.array([[age, finish_seconds]])
    X_scaled = scaler.transform(X)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    cluster_name = metadata['cluster_names'][cluster_id]
    return cluster_id, cluster_name


def build_features_for_model(age, gender, country, race, year, preprocess):
    """Replica el feature engineering del N03 exactamente."""
    feature_cols = preprocess['feature_cols']
    countries_to_keep = preprocess['countries_to_keep']
    race_encoding_map = preprocess['race_encoding_map']
    race_country_map = preprocess['race_country_map']
    mode_category = preprocess['mode_category']
    global_mean_bq = preprocess['global_mean_bq']

    features = {col: 0 for col in feature_cols}
    features['Year'] = year
    features['Age'] = age
    features['Age_Squared'] = age ** 2
    features['Gender_M'] = 1 if gender == 'M' else 0

    if country in countries_to_keep:
        col = f'Country_{country}'
        if col in features:
            features[col] = 1
    else:
        if 'Country_Other' in features:
            features['Country_Other'] = 1

    race_country = race_country_map.get(race, 'US')
    features['Is_Home_Country'] = 1 if country == race_country else 0

    if race in race_encoding_map.index:
        features['Race_te'] = float(race_encoding_map[race])
    else:
        features['Race_te'] = global_mean_bq

    features['Race_Category_enc'] = mode_category

    return pd.DataFrame([features])[feature_cols]


def get_threshold_for_age(age):
    if 30 <= age <= 39:
        return THRESHOLD_30_39
    return THRESHOLD_DEFAULT


# =====================================================================
# UI
# =====================================================================

# Hero image
st.image(
    str(BASE_DIR / 'app_streamlit' / 'assets' / 'boston_marathon.jpg'),
    use_container_width=True,
)

# Logo + título en la misma línea
col_logo, col_title, col_spacer = st.columns([1, 2, 1], vertical_alignment='center')

with col_logo:
    st.image(
        str(BASE_DIR / 'app_streamlit' / 'assets' / 'boston_logo.png'),
        width=600,
    )

with col_title:
    st.markdown(
        '<div style="text-align: center;">'
        '<h1 style="color: white; margin-bottom: 0;">Boston BQ Predictor</h1>'
        '<p style="color: #FFC72C; font-size: 1.1em; font-weight: 300; margin-top: 5px;">'
        'Tu probabilidad de cualificar para Boston, basada en datos de 1M+ finishers.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# col_spacer queda vacía a propósito para que el título quede centrado en pantalla

try:
    art = load_artifacts()
except FileNotFoundError as e:
    st.error(f'No se pudo cargar un artefacto: {e}')
    st.stop()

RACES_AVAILABLE = sorted(art['preprocess']['race_encoding_map'].index.tolist())
COUNTRIES_AVAILABLE = sorted(art['preprocess']['countries_to_keep']) + ['Other']

# ---------- 1. INPUT ----------

st.markdown('---')
st.subheader('1. Cuéntame sobre ti')

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Edad', min_value=18, max_value=85, value=33, step=1)
    gender = st.radio('Género', ['M', 'F'], horizontal=True)
    country = st.selectbox(
        'País de origen',
        options=COUNTRIES_AVAILABLE,
        index=COUNTRIES_AVAILABLE.index('ES') if 'ES' in COUNTRIES_AVAILABLE else 0,
    )

with col2:
    finish_str = st.text_input(
        'Tu mejor tiempo de maratón (HH:MM:SS)',
        value='03:45:00',
    )
    race = st.selectbox(
        'Carrera donde corriste',
        options=RACES_AVAILABLE,
        index=RACES_AVAILABLE.index('Berlin Marathon') if 'Berlin Marathon' in RACES_AVAILABLE else 0,
    )
    year = st.number_input('Año de la carrera', min_value=2015, max_value=2026, value=2024, step=1)

try:
    finish_seconds = time_str_to_seconds(finish_str)
    if finish_seconds < 7200 or finish_seconds > 25200:
        st.warning('El tiempo está fuera del rango realista (2h-7h).')
except ValueError:
    st.error('Formato de tiempo inválido. Usa HH:MM:SS.')
    st.stop()

if country != 'US':
    st.info(
        'El modelo está entrenado mayoritariamente con datos de runners USA '
        '(~70%). Las predicciones para otros países pueden ser menos precisas.'
    )

with st.expander('Cómo interpretar la predicción'):
    st.markdown(
        '**El modelo predice tu probabilidad de BQ a priori**, basándose en tu '
        'perfil demográfico (edad, género, país) y la carrera donde corres. '
        'No usa tu tiempo personal como input.\n\n'
        '**Esto significa que:**\n'
        '- La predicción te dice qué probabilidad tendría un runner *como tú* '
        '(misma edad, género, país, carrera) de cualificar.\n'
        '- Si tu tiempo real ya es BQ pero el modelo dice "No BQ", significa que '
        'estás superando la expectativa de tu perfil demográfico.\n'
        '- Si tu tiempo real no llega al corte BQ pero el modelo dice "BQ", '
        'significa que tu perfil tiende a cualificar pero tú aún estás por debajo.\n\n'
        '**Tu tiempo se usa para asignar tu arquetipo de runner** (sección 3) '
        'y para verificar el cumplimiento BQ contra el corte oficial.'
    )

# ---------- BOTÓN ----------

if st.button('Calcular mi predicción', type='primary', use_container_width=True):

    # 1. Predicción supervisada
    X = build_features_for_model(age, gender, country, race, year, art['preprocess'])
    X = X[art['model_features']]
    proba = float(art['model'].predict_proba(X)[0, 1])
    threshold = get_threshold_for_age(age)
    is_bq = proba >= threshold

    # 2. Cluster
    cluster_id, cluster_name = predict_cluster(
        age, finish_seconds, art['scaler'], art['kmeans'], art['cluster_metadata']
    )

    # 3. Stats del cluster
    df_c = art['df_clusters']
    cluster_bq_rate = df_c[df_c['cluster_name'] == cluster_name]['es_BQ'].mean() * 100
    cluster_size_pct = (df_c['cluster_name'] == cluster_name).mean() * 100

    # ---------- 2. RESULTADO MODELO ----------

    st.markdown('---')
    st.subheader('2. Tu probabilidad de cualificar')

    col1, col2, col3 = st.columns(3)
    col1.metric('Probabilidad BQ', f'{proba * 100:.1f}%')
    col2.metric('Threshold', f'{threshold:.3f}', help=f'Ajustado por edad ({age} años).')
    col3.metric('Predicción', 'BQ' if is_bq else 'No BQ')

    if is_bq:
        st.success(
            f'Según el modelo, **clasificas para Boston** con una probabilidad de '
            f'{proba * 100:.1f}%.'
        )
    else:
        gap = (threshold - proba) * 100
        st.warning(
            f'Aún no clasificas según el modelo. Te faltan {gap:.1f} puntos de '
            f'probabilidad para alcanzar el threshold de {threshold:.3f}.'
        )

    # Verificación BQ directa contra el corte oficial
    bq_standard = get_bq_standard(age, gender)
    gap_to_bq = bq_standard - finish_seconds

    if gap_to_bq >= 0:
        st.info(
            f'**Verificación oficial BQ:** tu tiempo de {fmt_seconds(finish_seconds)} '
            f'**cumple** el corte BQ para tu categoría ({fmt_seconds(bq_standard)}). '
            f'Margen: {int(gap_to_bq // 60)} minutos por debajo del corte.'
        )
    else:
        st.info(
            f'**Verificación oficial BQ:** tu tiempo de {fmt_seconds(finish_seconds)} '
            f'**no cumple** el corte BQ ({fmt_seconds(bq_standard)}). '
            f'Te faltan {int(abs(gap_to_bq) // 60)} minutos.'
        )

    # ---------- 3. CLUSTER ----------

    st.markdown('---')
    st.subheader('3. Tu arquetipo de runner')

    centroid = art['cluster_metadata']['centroids_original'][cluster_id]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'### {cluster_name}')
        st.markdown(f'**{cluster_size_pct:.1f}%** del dataset')
        st.markdown(f'Tasa BQ del cluster: **{cluster_bq_rate:.1f}%**')

    with col2:
        st.markdown('**Perfil promedio del cluster:**')
        st.markdown(f'- Edad: {centroid["Age"]:.0f} años')
        st.markdown(f'- Tiempo: {fmt_seconds(centroid["Finish"])}')
        st.markdown('**Tu posición:**')
        diff_age = age - centroid['Age']
        diff_min = (finish_seconds - centroid['Finish']) / 60
        st.markdown(f'- Edad: {age} años ({diff_age:+.0f} vs media del cluster)')
        st.markdown(f'- Tiempo: {fmt_seconds(finish_seconds)} ({diff_min:+.1f} min vs media)')

    # ---------- 4. VISUALIZACIÓN ----------

    st.markdown('---')
    st.subheader('4. ¿Dónde estás en el mapa de runners?')

    palette = {
        'Joven Avanzado':     '#2A9D8F',
        'Veterano Avanzado':  '#264653',
        'Joven Aspirante':    '#E9C46A',
        'Veterano Aspirante': '#E76F51',
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    df_plot = df_c.sample(min(8000, len(df_c)), random_state=42)
    for name, color in palette.items():
        sub = df_plot[df_plot['cluster_name'] == name]
        ax.scatter(sub['Age'], sub['Finish'] / 3600,
                   c=color, alpha=0.25, s=8, label=name)

    ax.scatter(age, finish_seconds / 3600,
               c='red', s=400, marker='*',
               edgecolor='white', linewidth=2, zorder=10, label='TÚ')

    ax.set_xlabel('Edad')
    ax.set_ylabel('Tiempo de maratón (horas)')
    ax.set_title('Tu posición entre 1M+ runners')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ---------- 5. SUGERENCIA ----------

    st.markdown('---')
    st.subheader('5. ¿Cómo mejorar?')

    if 'Aspirante' in cluster_name:
        target_cluster = cluster_name.replace('Aspirante', 'Avanzado')
        target_id = [k for k, v in art['cluster_metadata']['cluster_names'].items()
                     if v == target_cluster][0]
        target_centroid = art['cluster_metadata']['centroids_original'][target_id]
        target_bq = df_c[df_c['cluster_name'] == target_cluster]['es_BQ'].mean() * 100
        gap_minutes = (finish_seconds - target_centroid['Finish']) / 60

        st.info(
            f'Para pasar al cluster **{target_cluster}** '
            f'(tasa BQ: **{target_bq:.1f}%**), necesitarías bajar tu tiempo unos '
            f'**{gap_minutes:.0f} minutos** hasta aproximadamente '
            f'**{fmt_seconds(target_centroid["Finish"])}**.'
        )
        st.caption(
            'Pasar de Aspirante a Avanzado es el salto cualitativo más grande '
            'en términos de probabilidad BQ (de ~1% a ~20%+). Suele requerir '
            '6-12 meses de entrenamiento estructurado.'
        )
    else:
        if cluster_bq_rate < 50:
            st.info(
                f'Estás en el cluster **{cluster_name}**, donde el '
                f'{cluster_bq_rate:.1f}% cualifica. Tu margen de mejora pasa por '
                f'bajar tu tiempo dentro del cluster.'
            )
        else:
            st.success('Estás en el cluster con mayor tasa BQ. ¡Sigue así!')

# ---------- FOOTER ----------

st.markdown('---')
st.caption(
    'Modelo XGBoost (N07) + KMeans clustering (N08). '
    'Threshold F0.5 = 0.748 (0.5 para edad 30-39). '
    'Limitaciones: dataset 70-80% USA, género binario M/F, '
    'no captura runners élite (<2h30).'
)