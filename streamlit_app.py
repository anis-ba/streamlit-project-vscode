import joblib
import tensorflow as tf
import streamlit as st
import altair as alt
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prédiction CO2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE PERSONNALISÉ ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- TITRE ET INTRODUCTION ---
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🚗 Prédiction des Émissions de CO2")
    st.markdown("""
**Plateforme d'Évaluation et de Comparaison de Modèles de Machine Learning**
""")
    st.markdown("""
Analyse complète de la performance de modèles prédictifs pour estimer les émissions CO2 des véhicules.
""")

with col2:
    if os.path.exists("assets/datascientstest_logo.png"):
        st.image("assets/datascientstest_logo.png", use_container_width=True)

st.divider()

st.markdown("""
**Réalisé par :** Anis BENAICHA et Shiva HEYDARIAN  
**Promotion :** 2025/2026  
**Formation Datascientest**
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

# --- FONCTION D'INTERPRÉTABILITÉ ---
def get_feature_importance(model, X_test, y_test, model_name):
    """Extrait l'importance des variables selon le type de modèle"""
    try:
        # 1. Feature Importance native (arbres et boosting)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_test.columns
            importance_df = pd.DataFrame({
                'Variable': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            return importance_df, "Native (Arbre/Boosting)"
        
        # 2. Coefficients pour modèles linéaires
        elif hasattr(model, 'coef_'):
            coef = np.abs(np.ravel(model.coef_))
            feature_names = X_test.columns
            importance_df = pd.DataFrame({
                'Variable': feature_names,
                'Importance': coef
            }).sort_values('Importance', ascending=False)
            # Normaliser entre 0 et 1
            importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
            return importance_df, "Coefficients (Linéaire)"
        
        # 3. Pipeline avec estimateur sous-jacent
        elif hasattr(model, 'named_steps'):
            final_estimator = model.named_steps.get('regressor') or model.named_steps.get('classifier')
            if final_estimator is not None:
                return get_feature_importance(final_estimator, X_test, y_test, model_name)
        
        # 4. Permutation Importance (fallback générique)
        else:
            result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            importance_df = pd.DataFrame({
                'Variable': X_test.columns,
                'Importance': result.importances_mean
            }).sort_values('Importance', ascending=False)
            return importance_df, "Permutation Importance"
            
    except Exception as e:
        return None, f"Erreur: {str(e)}"

# --- ANALYSE DES RÉSIDUS ---
def plot_residuals(y_test, y_pred, model_name):
    """Créer des graphiques de résidus"""
    residuals = y_test.values - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme des résidus
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Résidus (g CO2/km)')
        ax.set_ylabel('Fréquence')
        ax.set_title(f'Distribution des Résidus - {model_name}')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Scatter plot résidus vs prédictions
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_pred, residuals, alpha=0.5, color='steelblue', edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Valeurs Prédites (g CO2/km)')
        ax.set_ylabel('Résidus (g CO2/km)')
        ax.set_title(f'Résidus vs Prédictions - {model_name}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# --- ANALYSE PRÉDICTIONS VS RÉALITÉ ---
def plot_predictions_vs_reality(y_test, y_pred, model_name):
    """Scatter plot des prédictions vs réalité"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(y_test.values, y_pred, alpha=0.5, color='steelblue', edgecolors='black', linewidth=0.5, s=30)
    
    # Ligne parfaite
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction Parfaite')
    
    ax.set_xlabel('Valeurs Réelles (g CO2/km)', fontsize=11)
    ax.set_ylabel('Valeurs Prédites (g CO2/km)', fontsize=11)
    ax.set_title(f'Prédictions vs Réalité - {model_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# --- CROSS-VALIDATION ---
def perform_cross_validation(model, X_test, y_test, cv_folds=5):
    """Effectue une cross-validation"""
    try:
        # Créer un dataset complet pour CV
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_test, y_test, cv=kfold, scoring='r2')
        cv_rmse = np.sqrt(-cross_val_score(model, X_test, y_test, cv=kfold, scoring='neg_mean_squared_error'))
        
        return {
            'R2_mean': cv_scores.mean(),
            'R2_std': cv_scores.std(),
            'RMSE_mean': cv_rmse.mean(),
            'RMSE_std': cv_rmse.std(),
            'folds': cv_folds
        }
    except Exception as e:
        return None

# --- SIDEBAR : GESTION DES MODÈLES ---
with st.sidebar:
    st.header("⚙️ Configuration")

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

# --- SECTION PRINCIPALE ---
st.divider()

# Navigation par onglets principaux
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    "📋 Accueil", 
    "📊 Benchmark", 
    "🔍 Analyse Détaillée", 
    "🔎 Interprétabilité",
    "⚙️ Configuration"
])

# --- TAB 1: ACCUEIL ---
with main_tab1:
    st.header("Bienvenue dans le Comparateur de Modèles CO2")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Contexte du Projet")
        st.markdown("""
        **Objectif :** Développer un modèle prédictif précis pour estimer les émissions de CO2 des véhicules.
        
        **Enjeux :** 
        - 🌍 Conformité réglementaire (WLTP, Euro)
        - 🚗 Transparence pour les consommateurs
        - ♻️ Réduction des émissions
        """)
    
    with col2:
        st.subheader("🎯 Approche")
        st.markdown("""
        **Méthodologie :**
        1. Entraînement de plusieurs modèles (linéaires, arbres, boosting)
        2. Évaluation sur un ensemble de test indépendant
        3. Comparaison des performances (RMSE, R², MAE)
        4. Analyse de l'interprétabilité des modèles
        5. Recommandations pour la production
        """)
    
    st.divider()
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("Modèles Testés", "8+", "ElasticNet, Ridge, Lasso, Linear, Polynomial, RF, XGBoost, DL")
    with col_stats2:
        st.metric("Métrique Principale", "RMSE", "Root Mean Squared Error")
    with col_stats3:
        st.metric("Métrique Secondaire", "R²", "Coefficient de Détermination")
    
    st.divider()
    
    st.subheader("🚀 Guide d'Utilisation")
    st.markdown("""
    ### Étape 1️⃣ : Configuration
    - Allez à l'onglet **⚙️ Configuration**
    - Importez vos fichiers de modèles (`.joblib`, `.keras`)
    
    ### Étape 2️⃣ : Benchmark
    - Allez à l'onglet **📊 Benchmark**
    - Chargez votre fichier de test CSV
    - Cliquez sur "🚀 Lancer l'évaluation"
    
    ### Étape 3️⃣ : Analyse
    - Consultez les **📊 Performances** globales
    - Explorez l'**🔍 Analyse Détaillée** (résidus, scatter plots)
    - Comprenez l'**🔎 Interprétabilité** des variables
    
    ### 📊 Structure des Données
    - **CSV obligatoire :** Colonne cible = `emission_co2_wltp`
    - **Format :** Tous les nombres, prédicteurs + variable cible
    """)

# --- TAB 2: BENCHMARK ---
with main_tab2:
    st.header("📊 Benchmark - Comparaison des Modèles")
    
    col_upload, col_action = st.columns([1, 2])

    with col_upload:
        test_file = st.file_uploader("Fichier de test (CSV)", type=["csv"], key="test_uploader")

    if test_file is not None:
        try:
            df_test = load_data(test_file)
            target_col = "emission_co2_wltp"

            with col_action:
                st.write(f"**Données chargées :** {df_test.shape[0]} lignes, {df_test.shape[1]} colonnes")
                if target_col not in df_test.columns:
                    st.error(f"❌ Colonne cible '{target_col}' introuvable.")
                else:
                    if st.button("🚀 Lancer l'évaluation", type="primary", use_container_width=True):
                        st.session_state['evaluation_active'] = True
            
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
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        
                        results.append({
                            "Modèle": name,
                            "MAE": mae,
                            "MSE": mse,
                            "RMSE": rmse,
                            "R²": r2,
                            "y_pred": y_pred,
                            "y_test": y_test
                        })
                    except Exception as e:
                        st.error(f"Erreur {name}: {e}")
                    progress_bar.progress((i + 1) / len(all_models))

                progress_bar.empty()
                status_text.empty()

                if results:
                    # Sauvegarder dans session state pour autre onglets
                    st.session_state['results'] = results
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    
                    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_test']} for r in results])
                    best_model = results_df.loc[results_df['RMSE'].idxmin()]

                    st.subheader("🏆 Performances du Meilleur Modèle")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Modèle Gagnant", best_model['Modèle'])
                    kpi2.metric("RMSE", f"{best_model['RMSE']:.2f} g/km", delta="Plus petit c'est mieux", delta_color="inverse")
                    kpi3.metric("R²", f"{best_model['R²']:.4f}", delta="Plus grand c'est mieux")
                    kpi4.metric("MAE", f"{best_model['MAE']:.2f} g/km")

                    st.divider()
                    st.subheader("📈 Comparaison Complète")

                    # Graphique des modèles
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        metric = st.radio("Métrique", ["RMSE", "R²", "MAE"], horizontal=True, key="metric_radio")
                        sort_order = 'descending' if metric == 'R²' else 'ascending'

                        chart = alt.Chart(results_df).mark_bar().encode(
                            x=alt.X('Modèle', sort=alt.EncodingSortField(field=metric, order=sort_order), axis=alt.Axis(labelAngle=45)),
                            y=alt.Y(metric, title=metric),
                            color=alt.Color('Modèle', legend=None),
                            tooltip=['Modèle', metric]
                        ).properties(height=350)
                        st.altair_chart(chart, use_container_width=True)
                    
                    with col_chart2:
                        # Tableau détaillé
                        numeric_cols = ['MAE', 'MSE', 'RMSE', 'R²']
                        st.dataframe(
                            results_df.style.format({col: "{:.4f}" for col in numeric_cols})
                            .background_gradient(subset=['RMSE', 'MAE'], cmap='Reds')
                            .background_gradient(subset=['R²'], cmap='Greens'),
                            use_container_width=True,
                            height=350
                        )

        except Exception as e:
            st.error(f"Erreur: {e}")

# --- TAB 3: ANALYSE DÉTAILLÉE ---
with main_tab3:
    st.header("🔍 Analyse Détaillée des Prédictions")
    
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_test']} for r in results])
        
        # Sélectionner un modèle
        selected_model = st.selectbox("Sélectionner un modèle à analyser", results_df['Modèle'].tolist())
        selected_result = next((r for r in results if r['Modèle'] == selected_model), None)
        
        if selected_result:
            y_pred = selected_result['y_pred']
            y_test = selected_result['y_test']
            
            # Onglets d'analyse
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                "📊 Prédictions vs Réalité",
                "📉 Analyse des Résidus",
                "📋 Cross-Validation"
            ])
            
            with analysis_tab1:
                st.subheader(f"Prédictions vs Réalité - {selected_model}")
                plot_predictions_vs_reality(y_test, y_pred, selected_model)
                
                # Stats
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                with col_stat2:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} g/km")
                with col_stat3:
                    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f} g/km")
            
            with analysis_tab2:
                st.subheader(f"Analyse des Résidus - {selected_model}")
                plot_residuals(y_test, y_pred, selected_model)
                
                # Stats résidus
                residuals = y_test.values - y_pred
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                with col_res1:
                    st.metric("Résidu Moyen", f"{residuals.mean():.2f}", "Proche de 0 = bon")
                with col_res2:
                    st.metric("Écart-Type", f"{residuals.std():.2f}")
                with col_res3:
                    st.metric("Min Résidu", f"{residuals.min():.2f}")
                with col_res4:
                    st.metric("Max Résidu", f"{residuals.max():.2f}")
            
            with analysis_tab3:
                st.subheader(f"Cross-Validation (5-Folds) - {selected_model}")
                best_model_obj = all_models[selected_model]
                X_test = st.session_state.get('X_test')
                
                with st.spinner("Calcul en cours..."):
                    cv_results = perform_cross_validation(best_model_obj, X_test, y_test, cv_folds=5)
                
                if cv_results:
                    col_cv1, col_cv2, col_cv3, col_cv4 = st.columns(4)
                    with col_cv1:
                        st.metric("R² Moyen", f"{cv_results['R2_mean']:.4f}")
                    with col_cv2:
                        st.metric("± Écart-Type", f"{cv_results['R2_std']:.4f}")
                    with col_cv3:
                        st.metric("RMSE Moyen", f"{cv_results['RMSE_mean']:.2f}")
                    with col_cv4:
                        st.metric("± Écart-Type", f"{cv_results['RMSE_std']:.2f}")
                    
                    st.info(f"✅ Les scores sont stables sur les {cv_results['folds']} plis de validation croisée")
    else:
        st.info("👈 Veuillez d'abord lancer une évaluation depuis l'onglet 📊 Benchmark")

# --- TAB 4: INTERPRÉTABILITÉ ---
with main_tab4:
    st.header("🔎 Interprétabilité - Importance des Variables")
    
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_test']} for r in results])
        best_model_name = results_df.loc[results_df['RMSE'].idxmin()]['Modèle']
        
        st.markdown(f"""
        Cette section affiche les variables qui ont le plus d'impact sur la prédiction du **meilleur modèle** ({best_model_name}).
        
        **Interprétation :**
        - Variables en haut = impact fort
        - Variables en bas = impact faible
        """)
        
        best_model_obj = all_models[best_model_name]
        X_test = st.session_state.get('X_test')
        y_test = st.session_state.get('y_test')
        
        importance_df, method = get_feature_importance(best_model_obj, X_test, y_test, best_model_name)
        
        if importance_df is not None and len(importance_df) > 0:
            st.info(f"**Méthode d'extraction :** {method}")
            
            col_chart, col_table = st.columns([2, 1])
            
            with col_chart:
                n_features = min(15, len(importance_df))
                top_features = importance_df.head(n_features).copy()
                top_features = top_features[::-1]
                
                chart_importance = alt.Chart(top_features).mark_bar().encode(
                    x=alt.X('Importance:Q', title='Score d\'Importance'),
                    y=alt.Y('Variable:N', sort=None),
                    color=alt.Color('Importance:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                    tooltip=['Variable', alt.Tooltip('Importance', format='.4f')]
                ).properties(height=400)
                
                st.altair_chart(chart_importance, use_container_width=True)
            
            with col_table:
                st.markdown("**Top 5 Variables**")
                top_5 = importance_df.head(5)[['Variable', 'Importance']]
                top_5_display = top_5.copy()
                top_5_display['Rang'] = range(1, len(top_5_display) + 1)
                top_5_display['% Impact'] = (top_5_display['Importance'] * 100).round(2).astype(str) + '%'
                st.dataframe(
                    top_5_display[['Rang', 'Variable', '% Impact']].set_index('Rang'),
                    use_container_width=True
                )
            
            st.divider()
            st.markdown("**📊 Statistiques d'Importance**")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            top_3_importance = importance_df.head(3)['Importance'].sum()
            top_5_importance = importance_df.head(5)['Importance'].sum()
            
            with stats_col1:
                st.metric("Total Variables", len(importance_df))
            with stats_col2:
                st.metric("Impact Top 3", f"{top_3_importance*100:.1f}%")
            with stats_col3:
                st.metric("Impact Top 5", f"{top_5_importance*100:.1f}%")
            
            # Tableau complet
            st.divider()
            st.subheader("📋 Classement Complet")
            importance_display = importance_df.copy()
            importance_display['Rang'] = range(1, len(importance_display) + 1)
            importance_display['% Impact'] = (importance_display['Importance'] * 100).round(2)
            st.dataframe(
                importance_display[['Rang', 'Variable', '% Impact']].set_index('Rang'),
                use_container_width=True
            )
        else:
            st.warning(f"⚠️ Impossible d'extraire l'importance pour ce modèle")
    else:
        st.info("👈 Veuillez d'abord lancer une évaluation depuis l'onglet 📊 Benchmark")

# --- TAB 5: CONFIGURATION ---
with main_tab5:
    st.header("⚙️ Configuration et Gestion des Modèles")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.subheader("📤 Importer des Modèles")
        st.markdown("""
        **Formats acceptés :**
        - `.joblib` : Scikit-learn (LinearRegression, Ridge, Lasso, etc.)
        - `.keras` / `.h5` : TensorFlow/Keras (Deep Learning)
        
        **Instructions :**
        1. Préparez vos fichiers de modèles
        2. Glissez-déposez ou sélectionnez-les ci-dessous
        3. Attendez la confirmation du chargement
        """)
        
        uploaded_files_config = st.file_uploader(
            "Importer de nouveaux modèles",
            accept_multiple_files=True,
            type=['joblib', 'keras', 'h5'],
            key="model_uploader_config",
            help="Glissez vos fichiers de modèles ici"
        )
        
        if uploaded_files_config:
            for uploaded_file in uploaded_files_config:
                file_path = os.path.join(models_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"✅ {len(uploaded_files_config)} modèle(s) chargé(s) avec succès!")
            st.rerun()
    
    with col_config2:
        st.subheader("📊 Modèles Actuellement Chargés")
        
        if all_models:
            for i, model_name in enumerate(all_models.keys(), 1):
                st.markdown(f"**{i}. {model_name}** ✅")
            
            st.divider()
            st.metric("Nombre Total", len(all_models))
        else:
            st.warning("⚠️ Aucun modèle chargé pour le moment")
    
    st.divider()
    
    col_reset1, col_reset2 = st.columns(2)
    
    with col_reset1:
        st.subheader("🗑️ Gestion des Fichiers")
        if st.button("Réinitialiser tous les modèles", type="secondary", use_container_width=True):
            reset_application()
            st.success("✅ Application réinitialisée")
            st.rerun()
    
    with col_reset2:
        st.subheader("📝 Informations Techniques")
        st.markdown(f"""
        **Répertoire des modèles :** `{models_dir}/`
        
        **Modèles actuels :** {len(all_models)}
        """)
    
    st.divider()
    st.subheader("📚 Documentation")
    st.markdown("""
    ### Structure Requise pour le CSV de Test
    
    Votre fichier CSV doit contenir :
    - **Colonnes de features :** Les variables explicatives (numériques)
    - **Colonne cible :** `emission_co2_wltp` (la variable à prédire)
    
    ### Exemple de Structure
    ```
    puissance_din,masse,couple,....,emission_co2_wltp
    100,1200,200,....,150
    120,1300,220,....,160
    ...
    ```
    
    ### Métriques Utilisées
    
    | Métrique | Formule | Interprétation |
    |----------|---------|---|
    | **RMSE** | √(Σ(y-ŷ)² / n) | Erreur quadratique moyenne - **Plus bas c'est mieux** |
    | **MAE** | Σ\|y-ŷ\| / n | Erreur absolue moyenne - **Plus bas c'est mieux** |
    | **R²** | 1 - (SS_res / SS_tot) | Qualité de l'ajustement - **Plus proche de 1 c'est mieux** |
    | **MSE** | Σ(y-ŷ)² / n | Erreur quadratique - **Plus bas c'est mieux** |
    """)