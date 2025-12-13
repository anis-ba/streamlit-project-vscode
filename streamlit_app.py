#streamlit run  streamlit_app.py --server.maxUploadSize 900
import joblib
import tensorflow as tf
import streamlit as st
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # Import the standard scikit-learn RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Set the main title of the Streamlit application
st.title("Prédiction des Émissions de CO2 des Véhicules")

# Add a sidebar to the application
st.sidebar.title("Navigation et Paramètres")
st.sidebar.header("Configuration des Modèles")

# --- MODIFICATION START ---
# Création d'un dossier temporaire pour stocker les modèles uploadés
temp_dir = "temp_models"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Option 1 : Upload via l'interface (fonctionne partout)
st.sidebar.subheader("Charger les modèles")
uploaded_files = st.sidebar.file_uploader(
    "Déposez vos fichiers .joblib et .keras ici",
    accept_multiple_files=True,
    type=['joblib', 'keras', 'h5']
)

models_dir = temp_dir # Par défaut, on pointe vers le dossier temporaire

# Si des fichiers sont uploadés, on les sauvegarde localement
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"{len(uploaded_files)} fichiers chargés dans l'application.")
else:
    # Option 2 : Chemin local (si vous testez sur votre propre PC)
    models_dir = st.sidebar.text_input("Ou spécifiez un chemin local si vous exécutez en local:", temp_dir)
# --- MODIFICATION END ---

dl_model_filename = 'deep_learning_model.keras'

# Dictionnaire pour stocker les modèles chargés
models = {}

# Chargement de tous les modèles .joblib présents dans le dossier
if os.path.exists(models_dir):
    @st.cache_resource
    def load_joblib_model(path):
        return joblib.load(path)

    # Lister les fichiers et charger ceux avec l'extension .joblib
    files_in_dir = os.listdir(models_dir)
    if not files_in_dir and not uploaded_files:
        st.warning("Aucun fichier trouvé. Veuillez uploader vos modèles via la barre latérale.")

    for filename in files_in_dir:
        if filename.endswith('.joblib'):
            file_path = os.path.join(models_dir, filename)
            model_name = os.path.splitext(filename)[0]
            try:
                models[model_name] = load_joblib_model(file_path)
                st.success(f"Modèle '{model_name}' chargé avec succès.")
            except Exception as e:
                st.error(f"Erreur lors du chargement de {filename}: {e}")
else:
    st.error(f"Le dossier spécifié est introuvable : {models_dir}")

# Tentative de définir rf_pipeline
rf_pipeline = None
for name, model in models.items():
    if 'random_forest' in name.lower():
        rf_pipeline = model
        break

# Code Keras
@st.cache_resource
def load_dl_model(filename):
    return tf.keras.models.load_model(filename)

# Construction du chemin complet pour le modèle Keras
dl_model_path = os.path.join(models_dir, dl_model_filename)

# Vérification spécifique pour Keras
if os.path.exists(dl_model_path):
    try:
        dl_model = load_dl_model(dl_model_path)
        st.success("Deep Learning model loaded successfully!")
    except Exception as e:
         st.error(f"Erreur lors du chargement du modèle Deep Learning: {e}")
else:
    # On n'affiche l'avertissement que si l'utilisateur a déjà uploadé quelque chose
    if uploaded_files or os.listdir(models_dir):
        st.warning(f"Fichier modèle Deep Learning introuvable : {dl_model_filename}")

# Section pour la suppression des fichiers uploadés
st.sidebar.subheader("Suppression des fichiers")
if st.sidebar.button("Supprimer les fichiers uploadés"):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        st.sidebar.success("Fichiers supprimés avec succès.")
        # Recréer le dossier vide si nécessaire
        os.makedirs(temp_dir)
    else:
        st.sidebar.info("Aucun fichier à supprimer.")

# --- SECTION EVALUATION ---
st.divider()
st.header("Évaluation et Comparaison des Modèles")

# Upload du fichier de test
test_file = st.file_uploader("Chargez votre fichier CSV de test pour l'évaluation", type=["csv"])

if test_file is not None:
    try:
        df_test = pd.read_csv(test_file)
        st.write("Aperçu des données de test :")
        st.dataframe(df_test.head())

        # Sélection de la variable cible
        target_col = st.selectbox("Sélectionnez la colonne cible (vérité terrain)", df_test.columns)

        if st.button("Lancer l'évaluation"):
            X_test = df_test.drop(columns=[target_col])
            y_test = df_test[target_col]
            
            results = []

            # 1. Évaluation des modèles Scikit-learn / Joblib
            for name, model in models.items():
                try:
                    y_pred = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        "Modèle": name,
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2
                    })
                except Exception as e:
                    st.error(f"Erreur lors de l'évaluation du modèle {name}: {e}")

            # 2. Évaluation du modèle Deep Learning (si chargé)
            if 'dl_model' in locals() and dl_model is not None:
                try:
                    # Prédiction avec Keras (attention aux types de données)
                    # .flatten() est utilisé car Keras retourne souvent un tableau 2D (N, 1)
                    y_pred_dl = dl_model.predict(X_test).flatten()
                    
                    mae = mean_absolute_error(y_test, y_pred_dl)
                    mse = mean_squared_error(y_test, y_pred_dl)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred_dl)
                    
                    results.append({
                        "Modèle": "Deep Learning (Keras)",
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2
                    })
                except Exception as e:
                    st.error(f"Erreur lors de l'évaluation du modèle Deep Learning : {e}")

            # Affichage des résultats
            if results:
                results_df = pd.DataFrame(results)
                st.subheader("Tableau des performances")
                st.dataframe(results_df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}))
                
                # Identifier le meilleur modèle (basé sur le RMSE le plus bas)
                if not results_df.empty:
                    best_model_idx = results_df['RMSE'].idxmin()
                    best_model_row = results_df.loc[best_model_idx]
                    st.success(f"🏆 Le meilleur modèle est **{best_model_row['Modèle']}** avec un RMSE de **{best_model_row['RMSE']:.4f}**.")
            else:
                st.warning("Aucun modèle n'a pu être évalué correctement.")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
