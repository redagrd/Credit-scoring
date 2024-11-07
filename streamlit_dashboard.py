# streamlit_dashboard.py

import streamlit as st
import requests
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Titre du dashboard
st.title("Dashboard de Scoring de Crédit avec Suivi du Data Drift")

# URL de l'API pour les prédictions
api_url = "http://127.0.0.1:8000/predict"

# Demander l'identifiant client pour les prédictions
st.subheader("Prédiction de Crédit")
client_id = st.text_input("Entrez l'identifiant du client", value="100016")

# Bouton pour obtenir le score
if st.button("Obtenir le score de crédit"):
    if client_id:
        try:
            # Appel de l'API pour prédire
            response = requests.post(api_url, json={"client_id": int(client_id)})

            # Vérifier la réponse de l'API
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                
                # Afficher le résultat
                if prediction == 1:
                    st.success(f"Le client {client_id} est éligible pour un crédit.")
                else:
                    st.error(f"Le client {client_id} n'est pas éligible pour un crédit.")
            else:
                st.error(f"Erreur: {response.json()['detail']}")

        except Exception as e:
            st.error(f"Erreur lors de l'appel de l'API : {e}")
    else:
        st.warning("Veuillez entrer un identifiant client valide.")

# Section pour le suivi du Data Drift
st.subheader("Suivi du Data Drift")

# Fichiers de données
reference_data_path = "application_train.csv"  # Jeu de données de référence
new_data_path = "application_test.csv"  # Nouvelles données (simulées)

# Vérifier que les fichiers de données existent
if os.path.exists(reference_data_path) and os.path.exists(new_data_path):
    # Bouton pour générer et afficher le rapport de data drift
    if st.button("Afficher le rapport de Data Drift"):
        try:
            # Charger les données en ignorant les lignes problématiques
            reference_data = pd.read_csv(reference_data_path, encoding="utf-8", on_bad_lines="skip")
            new_data = pd.read_csv(new_data_path, encoding="utf-8", on_bad_lines="skip")

            # Spécifier les principales features à surveiller
            main_features = ['AMT_GOODS_PRICE', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_BIRTH'] 

            # Créer le dashboard de data drift avec DataDriftPreset
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=reference_data[main_features], current_data=new_data[main_features])

            # Sauvegarder le rapport dans un fichier HTML temporaire
            data_drift_report.save_html("data_drift_report.html")

            # Streamlit pour afficher le rapport
            st.title("Dashboard de Scoring de Crédit avec Suivi du Data Drift")

            # Afficher le rapport dans Streamlit
            with open("data_drift_report.html", "r") as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=1000, scrolling=True)

        except Exception as e:
            st.error(f"Erreur lors de la génération du rapport de Data Drift : {e}")
else:
    st.warning("Les fichiers de données de référence ou de nouvelles données sont introuvables.")





# # streamlit_dashboard.py

# import streamlit as st
# import requests

# # Titre du dashboard
# st.title("Dashboard de Scoring de Crédit")

# # Demande d'un identifiant client
# client_id = st.text_input("Entrez l'identifiant du client", value="100016")

# # API URL - Remplacer par l'URL de l'API si elle est déployée sur un serveur distant
# api_url = "http://127.0.0.1:8000/predict"

# # Bouton pour obtenir le score
# if st.button("Obtenir le score de crédit"):
#     if client_id:
#         try:
#             # Appel de l'API
#             response = requests.post(api_url, json={"client_id": int(client_id)})

#             # Vérifier la réponse de l'API
#             if response.status_code == 200:
#                 result = response.json()
#                 prediction = result["prediction"]
                
#                 # Afficher le résultat
#                 if prediction == 1:
#                     st.success(f"Le client {client_id} est éligible pour un crédit.")
#                 else:
#                     st.error(f"Le client {client_id} n'est pas éligible pour un crédit.")
#             else:
#                 st.error(f"Erreur: {response.json()['detail']}")

#         except Exception as e:
#             st.error(f"Erreur lors de l'appel de l'API : {e}")
#     else:
#         st.warning("Veuillez entrer un identifiant client valide.")
