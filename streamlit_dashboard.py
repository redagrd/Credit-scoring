import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Titre du dashboard
st.title("Dashboard de Scoring de Crédit")

# URL de l'API pour les prédictions
api_url = "https://creditrg-37c727617d30.herokuapp.com/predict"

# Charger les données globales
@st.cache_data
def load_data():
    data = pd.read_csv('final_processed_data_with_ids.csv')
    return data

data = load_data()

# Section pour les prédictions
st.subheader("Prédiction de Crédit")

# Demander l'identifiant client
client_id_input = st.text_input("Entrez l'identifiant du client :", value="100007")

# Initialiser l'état de l'application
if 'client_id' not in st.session_state:
    st.session_state['client_id'] = None
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'modified_features' not in st.session_state:
    st.session_state['modified_features'] = {}

# Fonction pour obtenir le score de crédit
def get_credit_score():
    client_id = client_id_input
    st.session_state['client_id'] = client_id
    try:
        # Préparer les données à envoyer à l'API
        payload = {"client_id": int(client_id)}
        if st.session_state.get('modified_features'):
            payload["modified_features"] = st.session_state['modified_features']

        # Appel de l'API pour obtenir la prédiction
        response = requests.post(api_url, json=payload)

        # Vérifier la réponse de l'API
        if response.status_code == 200:
            st.session_state['result'] = response.json()
        else:
            st.error(f"Erreur : {response.json()['detail']}")
            st.session_state['result'] = None
    except Exception as e:
        st.error(f"Erreur lors de l'appel de l'API : {e}")
        st.session_state['result'] = None

# Bouton pour obtenir le score de crédit
if st.button("Obtenir le score de crédit"):
    # Réinitialiser les features modifiées
    st.session_state['modified_features'] = {}
    get_credit_score()

# Vérifier si un résultat est disponible
if st.session_state['result'] is not None:
    result = st.session_state['result']
    client_id = st.session_state['client_id']
    prediction = result["prediction"]
    score = result.get("score", "N/A")
    client_features = pd.DataFrame([result["features"]])

    # Exclure 'skidcurr' des features affichées
    if 'skidcurr' in client_features.columns:
        client_features = client_features.drop(columns=['skidcurr'])

    # Afficher le résultat avec plus de détails
    if prediction == 1:
        st.success(f"Le client {client_id} est éligible pour un crédit.")
        if score != "N/A":
            st.info(f"Score du client : {score:.2f} (plus le score est élevé, plus le client est fiable)")
    else:
        st.error(f"Le client {client_id} n'est pas éligible pour un crédit.")
        if score != "N/A":
            st.info(f"Score du client : {score:.2f} (plus le score est bas, plus le client est risqué)")

    # Ajouter une jauge pour visualiser le score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,  # Convertir le score en pourcentage
        title={'text': "Score de Crédit (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))

    st.plotly_chart(fig)

    # Afficher les informations du client sans 'skidcurr'
    st.write("### Informations du client")
    st.dataframe(client_features.T)

    # Ajouter des champs pour modifier certaines features
    st.write("### Modifier les features du client")

    # Initialiser 'modified_features' dans st.session_state si nécessaire
    if 'modified_features' not in st.session_state:
        st.session_state['modified_features'] = {}

    # Sélection des features à modifier
    features_to_modify = {
        "daysbirth": "Âge (en années)",
        "extsource1": "Revenus externes 1",
        "extsource2": "Revenus externes 2",
        "extsource3": "Revenus externes 3",
        "amtincometotal": "Valeur totale des biens"
    }

    for feature_key, feature_label in features_to_modify.items():
        if feature_key in client_features.columns:
            current_value = client_features[feature_key].values[0]
            if feature_key == "daysbirth":
                # Conversion de l'âge en années positives pour l'affichage
                current_age = int(-current_value / 365)
                new_age = st.number_input(f"{feature_label} :", value=current_age, min_value=18, max_value=100, key=feature_key)
                if new_age != current_age:
                    # Convertir l'âge en jours négatifs pour le modèle
                    st.session_state['modified_features'][feature_key] = -new_age * 365
            else:
                new_value = st.number_input(f"{feature_label} :", value=float(current_value), key=feature_key)
                if new_value != current_value:
                    st.session_state['modified_features'][feature_key] = new_value

    # Bouton pour recalculer le score avec les features modifiées
    if st.button("Recalculer le score avec les features modifiées"):
        get_credit_score()

    # Liste des variables sans 'skidcurr'
    variables = [col for col in client_features.columns if col != 'skidcurr']

    # Sélectionner une variable pour la comparaison
    selected_variable = st.selectbox("Sélectionnez une variable pour la comparaison", variables, key='selected_variable')

    if selected_variable:
        # Distribution globale
        fig = px.histogram(data, x=selected_variable, nbins=50,
                           title=f'Distribution de {selected_variable} (Tous les clients)')
        client_value = client_features[selected_variable].values[0]
        fig.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="red",
                      annotation_text="Valeur du client", annotation_position="top left")
        st.plotly_chart(fig)

        # Filtrer les clients similaires
        filter_variable = st.selectbox("Sélectionnez une variable pour filtrer les clients similaires", variables, key='filter_variable')

        if filter_variable:
            unique_values = data[filter_variable].unique()
            selected_filter_value = st.selectbox(f"Sélectionnez une valeur pour {filter_variable}", unique_values, key='selected_filter_value')

            similar_clients = data[data[filter_variable] == selected_filter_value]

            # Distribution pour les clients similaires
            fig = px.histogram(similar_clients, x=selected_variable, nbins=50,
                               title=f'Distribution de {selected_variable} pour {filter_variable} = {selected_filter_value}')
            fig.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="red",
                          annotation_text="Valeur du client", annotation_position="top left")
            st.plotly_chart(fig)

    # Afficher l'importance locale et globale des features
    st.write("### Importance des features")

    # Importance locale
    local_importances_df = pd.DataFrame(result["local_importances"])
    local_importances_df['abs_shap_value'] = local_importances_df['shap_value'].abs()
    top_local_features = local_importances_df.sort_values('abs_shap_value', ascending=False).head(10)

    fig = px.bar(top_local_features, x='shap_value', y='feature', orientation='h',
                 title='Top 10 des features (Importance Locale)')
    st.plotly_chart(fig)

    # Importance globale
    global_importances_df = pd.DataFrame(result["global_importances"])
    top_global_features = global_importances_df.sort_values('importance', ascending=False).head(10)

    fig = px.bar(top_global_features, x='importance', y='feature', orientation='h',
                 title='Top 10 des features (Importance Globale)')
    st.plotly_chart(fig)

    # Comparaison des importances
    st.write("### Comparaison des importances locales et globales")

    # Fusionner les données locales et globales
    combined_importances = pd.merge(top_local_features, global_importances_df, on='feature', how='inner')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combined_importances['shap_value'],
        y=combined_importances['feature'],
        orientation='h',
        name='Importance Locale'
    ))
    fig.add_trace(go.Bar(
        x=combined_importances['importance'],
        y=combined_importances['feature'],
        orientation='h',
        name='Importance Globale'
    ))

    fig.update_layout(barmode='group', title='Comparaison des importances locales et globales')
    st.plotly_chart(fig)
else:
    if not client_id_input:
        st.warning("Veuillez entrer un identifiant client valide.")

