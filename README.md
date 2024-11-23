## Contexte du projet

L'objectif de ce projet est de construire un modèle de scoring de la probabilité de défaut de paiement d'un client pour un prêt bancaire.

Une API de ce modèle à été déployé sur le cloud, et à permis de créer un dashboard interactif pour présenter les résultats.

Voici le lien de l'api : https://creditrg-37c727617d30.herokuapp.com/predict

Pour tester l'API, vous pouvez utiliser la commande suivante dans un terminal :
```bash
curl -X POST "https://creditrg-37c727617d30.herokuapp.com/predict" -H "Content-Type: application/json" -d '{"client_id": 100007}'
```

Lien vers le dashboard streamlit utilisant le modèle de scoring de crédit :
https://redagrd-credit-scoring-streamlit-dashboard-shclty.streamlit.app/