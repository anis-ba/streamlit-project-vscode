import joblib
import tensorflow as tf
import streamlit as st
import os
import shutil
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # Import the standard scikit-learn RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


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
    st.sidebar.info("Ou spécifiez un chemin local si vous exécutez en local:")
    # Le chemin par défaut est vide pour éviter les erreurs Colab
    models_dir = st.sidebar.text_input("Chemin du dossier local", temp_dir)
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