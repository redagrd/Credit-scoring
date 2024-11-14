# streamlit_dashboard.py

import streamlit as st
import requests
import pandas as pd

# Titre du dashboard
st.title("Dashboard de Scoring de Crédit")

# URL de l'API pour les prédictions
api_url = "https://creditrg-37c727617d30.herokuapp.com/predict"

# Section pour les prédictions
st.subheader("Prédiction de Crédit")

# Demander l'identifiant client
client_id = st.text_input("Entrez l'identifiant du client :", value="100007")

# Bouton pour obtenir le score
if st.button("Obtenir le score de crédit"):
    if client_id:
        try:
            # Appel de l'API pour obtenir la prédiction
            response = requests.post(api_url, json={"client_id": int(client_id)})

            # Vérifier la réponse de l'API
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                score = result.get("score", "N/A")  # Si disponible, récupérer le score

                # Afficher le résultat avec plus de détails
                if prediction == 1:
                    st.success(f"Le client {client_id} est éligible pour un crédit.")
                    if score != "N/A":
                        st.info(f"Score du client : {score:.2f} (plus le score est élevé, plus le client est fiable)")
                else:
                    st.error(f"Le client {client_id} n'est pas éligible pour un crédit.")
                    if score != "N/A":
                        st.info(f"Score du client : {score:.2f} (plus le score est bas, plus le client est risqué)")
            else:
                st.error(f"Erreur : {response.json()['detail']}")

        except Exception as e:
            st.error(f"Erreur lors de l'appel de l'API : {e}")
    else:
        st.warning("Veuillez entrer un identifiant client valide.")

# Section pour afficher les informations générales
st.subheader("Informations supplémentaires")

st.write(
    """
    Ce dashboard permet de :
    - **Évaluer l'éligibilité d'un client** à un crédit en fonction de ses caractéristiques.
    - **Obtenir un score détaillé**, représentant la probabilité de remboursement du crédit.
    
    ### Méthodologie
    - Le modèle utilisé est un `LightGBM Classifier`.
    - Les prédictions sont effectuées à l'aide d'une API hébergée sur Heroku.
    - Le score affiché est la probabilité prédite pour la classe positive (client éligible).
    """
)

st.write("### Exemple d'ID Client")
st.write(
    """
    Voici des exemples d'IDs clients que vous pouvez utiliser pour tester le système :
    - **100007** : Client éligible
    - **100006** : Client non éligible
    """
)

# Section pour uploader un fichier CSV
st.subheader("Tester plusieurs clients à la fois")

uploaded_file = st.file_uploader("Chargez un fichier CSV avec des identifiants clients", type=["csv"])

if uploaded_file is not None:
    try:
        # Lire le fichier CSV
        client_data = pd.read_csv(uploaded_file)

        # Vérifier si la colonne client_id est présente
        if "client_id" in client_data.columns:
            # Créer une liste des IDs à tester
            ids = client_data["client_id"].tolist()

            # Envoyer les requêtes pour chaque ID
            results = []
            for client_id in ids:
                response = requests.post(api_url, json={"client_id": int(client_id)})
                if response.status_code == 200:
                    result = response.json()
                    results.append(
                        {
                            "client_id": client_id,
                            "prediction": result["prediction"],
                            "score": result.get("score", "N/A"),
                        }
                    )
                else:
                    results.append(
                        {
                            "client_id": client_id,
                            "prediction": "Erreur",
                            "score": "N/A",
                        }
                    )

            # Afficher les résultats sous forme de tableau
            results_df = pd.DataFrame(results)
            st.write("### Résultats des prédictions")
            st.dataframe(results_df)

            # Télécharger les résultats au format CSV
            st.download_button(
                label="Télécharger les résultats",
                data=results_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.error("Le fichier CSV doit contenir une colonne nommée 'client_id'.")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
