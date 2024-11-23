Lien vers le dashboard streamlit utilisant le modèle de scoring de crédit :
https://redagrd-credit-scoring-streamlit-dashboard-shclty.streamlit.app/

Lien vers l'api hébergée sur Heroku :
https://creditrg-37c727617d30.herokuapp.com/predict

Pour tester l'API, vous pouvez utiliser la commande suivante :
```bash
curl -X POST "https://creditrg-37c727617d30.herokuapp.com/predict" -H "Content-Type: application/json" -d '{"client_id": 100007}'
```