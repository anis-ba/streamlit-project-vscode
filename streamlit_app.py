import joblib
import tensorflow as tf
import streamlit as st
import altair as alt
import os
import shutil
import pandas as pd
import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prédiction CO2",
    page_icon="🚗",
    layout="wide"
)

# --- TITRE ET INTRODUCTION ---
st.title="🚗 Prédiction des Émissions de CO2"
st.markdown("""
**Comparateur de Performance de Modèles de Machine Learning**""")
st.markdown("""
Cette application permet d'évaluer et de comparer automatiquement plusieurs modèles prédictifs sur un jeu de données de test.
""")

# --- SETUP ET UTILITAIRES ---
models_dir = "temp_models"
dl_model_filename = 'deep_learning_model.keras'

if "startup_cleaned" not in st.session_state:
    if os.path.exists(models_dir):
        try: shutil.rmtree(models_dir)
        except: pass
    st.session_state["startup_cleaned"] = True

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def reset_application():
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
        os.makedirs(models_dir)
    for key in ["model_uploader", "test_uploader"]:
        if key in st.session_state:
            del st.session_state[key]

@st.cache_resource
def load_joblib_model(path):
    return joblib.load(path)

@st.cache_resource
def load_dl_model(path):
    return tf.keras.models.load_model(path)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# --- SIDEBAR : GESTION DES MODÈLES ---
with st.sidebar:
    st.header="⚙️ Configuration"

    with st.expander("ℹ️ Mode d'emploi", expanded=False):
        st.markdown("""
        1. **Déposez vos modèles** ci-dessous (.joblib, .keras).
        2. **Chargez un fichier CSV** contenant les features et la colonne cible.
        3. **Lancez l'évaluation** pour voir le classement.
        """)

    uploaded_files = st.file_uploader(
        "Importer des modèles",
        accept_multiple_files=True,
        type=['joblib', 'keras', 'h5'],
        key="model_uploader",
        help="Glissez vos fichiers de modèles ici"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(models_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} modèles chargés.")

    st.divider()
    st.button("🗑️ Réinitialiser l'application", on_click=reset_application, type="primary")

# --- CHARGEMENT DES MODÈLES ---
all_models = {}
if os.path.exists(models_dir):
    files_in_dir = os.listdir(models_dir)

    # 1. Scikit-learn
    for filename in files_in_dir:
        if filename.endswith('.joblib'):
            try:
                all_models[os.path.splitext(filename)[0]] = load_joblib_model(os.path.join(models_dir, filename))
            except Exception: pass

    # 2. Keras
    dl_path = os.path.join(models_dir, dl_model_filename)
    if os.path.exists(dl_path):
        try:
            all_models["Deep Learning (Keras)"] = load_dl_model(dl_path)
        except Exception: pass

    if all_models:
        st.sidebar.info(f"🧠 {len(all_models)} modèles prêts")
    elif not uploaded_files:
        st.info("👈 Veuillez commencer par charger vos modèles dans la barre latérale.")

# --- SECTION ÉVALUATION ---
st.divider()
st.header="📊 Benchmark"

col_upload, col_action = st.columns([1, 2])

with col_upload:
    test_file = st.file_uploader("Fichier de test (CSV)", type=["csv"], key="test_uploader")

if test_file is not None:
    try:
        df_test = load_data(test_file)
        target_col = "emission_CO2_WLTP"

        with col_action:
            st.write(f"**Données chargées :** {df_test.shape[0]} lignes, {df_test.shape[1]} colonnes")
            if target_col not in df_test.columns:
                st.error(f"❌ Colonne cible '{target_col}' introuvable.")
            else:
                # 1. On vérifie si le bouton est cliqué et on met à jour l'état
                if st.button("🚀 Lancer l'évaluation", type="primary", use_container_width=True):
                    st.session_state['evaluation_active'] = True
        
        # 2. On vérifie la variable de session au lieu du bouton directement
        if target_col in df_test.columns and st.session_state.get('evaluation_active', False):
            X_test = df_test.drop(columns=[target_col])
            y_test = df_test[target_col]
            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, model) in enumerate(all_models.items()):
                status_text.text(f"Évaluation de : {name}...")
                try:
                    y_pred = np.ravel(model.predict(X_test))
                    results.append({
                        "Modèle": name,
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "R²": r2_score(y_test, y_pred)
                    })
                except Exception as e:
                    st.error(f"Erreur {name}: {e}")
                progress_bar.progress((i + 1) / len(all_models))

            progress_bar.empty()
            status_text.empty()

            if results:
                results_df = pd.DataFrame(results)

                # --- AFFICHAGE DES RÉSULTATS ---
                best_model = results_df.loc[results_df['RMSE'].idxmin()]

                st.subheader("🏆 Performances du Meilleur Modèle")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Modèle Gagnant", best_model['Modèle'])
                kpi2.metric("RMSE (Erreur)", f"{best_model['RMSE']:.4f}", delta="Plus petit c'est mieux.", delta_color="inverse")
                kpi3.metric("R² (Score)", f"{best_model['R²']:.4f}")

                st.divider()

                # Vues détaillées
                tab1, tab2, tab3 = st.tabs(["📈 Graphiques Comparatifs", "📄 Tableau Détaillé", "🔍 Données Brutes"])

                with tab1:
                    metric = st.radio("Métrique à visualiser", ["RMSE", "R²", "MAE"], horizontal=True)
                    sort_order = 'descending' if metric == 'R²' else 'ascending'

                    chart = alt.Chart(results_df).mark_bar().encode(
                        x=alt.X('Modèle', sort=alt.EncodingSortField(field=metric, order=sort_order), axis=alt.Axis(labelAngle=-0)),
                        y=alt.Y(metric, title=metric),
                        color=alt.Color('Modèle', legend=None),
                        tooltip=['Modèle', metric]
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

                with tab2:
                    # Formatage uniquement des colonnes numériques pour éviter l'erreur sur le texte
                    numeric_cols = ['MAE', 'MSE', 'RMSE', 'R²']
                    st.dataframe(
                        results_df.style.format({col: "{:.4f}" for col in numeric_cols})
                        .background_gradient(subset=['RMSE', 'MAE'], cmap='Reds')
                        .background_gradient(subset=['R²'], cmap='Greens'),
                        use_container_width=True
                    )

                with tab3:
                    st.write("Aperçu des données de test", df_test.head())

    except Exception as e:
        st.error(f"Erreur lecture CSV : {e}")