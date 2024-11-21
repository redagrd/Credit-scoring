# test_main.py

import requests

# URL de l'API déployée
API_URL = "https://creditrg-37c727617d30.herokuapp.com"

def test_predict_valid_client():
    """Tester l'API avec un client existant pour vérifier la prédiction."""
    response = requests.post(f"{API_URL}/predict", json={"client_id": 100007})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data  # Vérifie si la réponse contient le champ 'prediction'
    assert data["prediction"] in [0, 1]  # Vérifie si la prédiction est 0 ou 1

def test_predict_nonexistent_client():
    """Tester l'API avec un client non-existant pour vérifier la gestion des erreurs."""
    response = requests.post(f"{API_URL}/predict", json={"client_id": 999999})

    # Vérifiez si le statut est 404 ou 500
    assert response.status_code in [404, 500]

    # Vérifiez le contenu du message d'erreur
    if response.status_code == 404:
        assert response.json()["detail"] == "Client ID not found"
    elif response.status_code == 500:
        assert "Erreur interne" in response.json()["detail"]


def test_predict_missing_client_id():
    """Tester l'API avec des données incorrectes pour vérifier la validation d'entrée."""
    response = requests.post(f"{API_URL}/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity pour un identifiant manquant

def test_api_handles_internal_error():
    """Tester si l'API gère correctement les erreurs internes."""
    # Envoyer une requête incorrecte pour forcer une erreur 500
    response = requests.post(f"{API_URL}/predict", json={"invalid_key": "value"})

    # Vérifiez si l'API renvoie un code de statut 500
    assert response.status_code == 500

    # Vérifiez si un message d'erreur est présent
    assert "detail" in response.json()
