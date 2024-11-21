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
    assert response.status_code == 404
    assert response.json()["detail"] == "Client ID not found"

def test_predict_missing_client_id():
    """Tester l'API avec des données incorrectes pour vérifier la validation d'entrée."""
    response = requests.post(f"{API_URL}/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity pour un identifiant manquant
