venv\Scripts\Activate.ps1



Explique moi comment définir mon scoring avec pondération des faux positifs et des faux négatifs.

kaggle a utiliser pour initialiser mon scoring, données des clients en entrée et en sortie on a le scoring, si positif le client a un crédit, si négatif le client n'a pas de crédit.

petit dashboard streamlit pour exposer les prédictions

J'ai mis en place l'environnement mlflow pour suivre les expériences et les modèles.
J'ai lancé une première expérience avec le modèle kaggle de lightgbm que j'ai essayé d'adapté à mon environnement
J'ai commencé à mettre un place une API flask pour exposer les prédictions

Et ensuite ?
Je dois déployer l'api sur un cloud, aurait tu des recommandations pour cela ?

Heroku permet de déployer

Combien de features dois je mettre dans mon modèle ? y a t il des informations sur les guidelines à suivre pour cela ?

Pour l'instant j'utilise 50 features et uniquement les fichier app_train/test, faut il utiliser tous les fichiers et les joindre ensemble ?

Features à prendre en compte : 
backword forward selection = valeur p supérieur à 0.05
Jointure entre toutes les tables

utiliser gridsearch pour trouver les meilleurs hyperparamètres et ne prendre que les features les plus importantes en entrée pour l'api flask
utiliser streamlit pour exposer les prédictions et entrer les features en entrée
envoyer l'app sur heroku

j'ai regardé le webinairte deployer une API de prédiction
cela va m'aider pour la partie streamlit
Il faut que je regarde des tutos pour heroku
Il faut aussi que je regarde des tutos pour le versionning de mon code avec git

pas besoin d'aller très loin = 1 set qui passe et un qui passe pas

Render pour deployer l'api sur le cloud

quand push ou pull il faut que ça déploi sur heroku ou render

taiter les faux positifs et faux négatifs pour le scoring donc ne pas prendre l'accuracy principalement

evidently ai pour le scoring sous forme de dashboard avec les faux positifs et faux négatifs et les features les plus importantes pour le modèle sur streamlit
cas particulier = id client, mais si ya le temps = rentrer des informations sur le client et avoir le scoring

ext = salaire supplémentaire des foyer a voir les features les plus importantes
data drift = si les données changent, le modèle doit être réentrainé


Il me faut maintenant mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API. J'aimerais utiliser fast API ici. Il faudrait par la suite que je crée un dashboard Streamlit dans lequel je puisse appeler un id client qui me permettrait de définir si ce client à le droit ou non à un crédit, avec une réponse binaire donc, à l'aide de l'api et de mon modèle. Il faudrait ensuite que je test l’utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production sur mes principales features

curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"amtannuity\": 3, \"amtcredit\":  2, \"amtgoodsprice\":  600, \"amtincometotal\":  50000, \"amtreqcreditbureauday\": 100}"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body '{"amtannuity": 3, "amtcredit": 2, "amtgoodsprice": 600, "amtincometotal": 50000, "amtreqcreditbureauday": 100}'

utiliser sql light pour récupérer les données facilement
plus utile que csv

github pour le versionning et les tests et push automatique sur streamlit et sur heroku ou render

flag = meilleur model pour aller chercher le dernier model en production
attention aux limitations de poids du model

push et aller sur streamlit et l'api

python.env pour les variables d'environnement pour les clés api pour le cloud et eviter de modifier le code


pour lancer l'api : uvicorn main:app --reload


Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body '{"client_id": 100007}'


Tout à fonctionné, il semblerait que mon modèle le plus performant est le light gbm. 
Il me faut maintenant mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API. J'aimerais utiliser fast API ici. Il faudrait par la suite que je crée un dashboard Streamlit dans lequel je puisse appeler un id client qui me permettrait de définir si ce client à le droit ou non à un crédit, avec une réponse binaire donc, à l'aide de l'api et de mon modèle. Il faudrait ensuite que je test l’utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production sur mes principales features.
Pourrait tu m'aider avec ces différentes étapes ?


Fusionner les branches et versionner :

Lorsque votre travail est stable, vous pouvez fusionner la branche dev dans main :
bash
Copier le code
git checkout main
git merge dev
git push origin main


Ajouter l'estimation en pourcentage dans le dashboard
passer une matrice plutot qu'un vecteur

plutot aller chercher les featrure importance locale pour le seamlit plutot que le datadrift

mettre l'api et le fichier de données sur le cloud et deployer le streamlit

mettre qu'une partie du fichier pour limiter la taille du fichier dans le cloud

faire une matrice de confusion pour les faux positifs et faux négatifs (important)

Il faudrait maintenant que j'implémente tout cela dans le cloud, pour pouvoir permettre à n'importe qui de faire appel à mon api et à mon modèle de n'importe ou pour ce projet. Comment faire ? Pour information, j'ai donc un model nommé 'lightgbm_model.pkl', un fichier avec mon code api nommé 'main.py', un fichier avec les données pour récupérer les identifiants et les associées aux données appropriées nommé 'final_processed_data_with_ids.csv'. J'ai aussi mon dashboard streamlit nommé 'streamlit_dashboard.py', mais je ne sais pas si je dois le mettre sur le cloud ou si je peux le déployer séparément via streamlit. Dans tout les cas, pourrait tu m'expliquer la marche à suivre pour que tout fonctionne sur le cloud ? Je souhaiterai essayer d'utiliser python anywhere pour l'outil de cloud si possible

curl -X POST "https://redagrd.pythonanywhere.com/predict" -H "Content-Type: application/json" -d "{\"client_id\": 100007}"

minimiser le requiremments.txt à scikit learn et fastapi (pas besoin de plus)
implémenter le test unitaire pour le workflow du modèle
mettre en place un pipeline de CI/CD pour le déploiement automatique sur le cloud avec github actions

heroku git:remote -a creditrg

curl -X POST "https://creditrg-37c727617d30.herokuapp.com/predict" -H "Content-Type: application/json" -d "{\"client_id\": 100007}"
-H "Content-Type: application/json" \
-d '{"client_id": 100007}'


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Chemins vers les fichiers nécessaires
MODEL_PATH = os.getenv("MODEL_PATH", "lightgbm_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "final_processed_data_with_ids.csv")
FEATURES_PATH = os.getenv("FEATURES_PATH", "model_features.txt")

# Charger le modèle
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Le fichier modèle '{MODEL_PATH}' est introuvable.")

# Charger les données
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
else:
    raise RuntimeError(f"Le fichier données '{DATA_PATH}' est introuvable.")

# Charger les features
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r') as f:
        model_features = f.read().splitlines()
else:
    raise RuntimeError(f"Le fichier des features '{FEATURES_PATH}' est introuvable.")

# Nom de la colonne d'identifiant client
client_id_col = "skidcurr"

# Initialiser l'application FastAPI
app = FastAPI()

# Modèle de requête pour FastAPI
class ClientIDRequest(BaseModel):
    client_id: int

coder la suite du straemlit et ajouter les graph
améliorer le model et le score avec les fp et fn

mettre une jauge pour le scoring
applique les test unitaires pour le modèle et l'api et utiliser github actions pour le CI/CD et le wprkflow général
utiliser le code de github
copier le contenu et créer un sous repertoire .github, un sous repertoire workflow et mettre le contenu
pour avoir le commit en local
ou sinon faire le commit en vert et voir ce qui se passe au niveau des actions
supprimer le commit fait en local, récupérer l'état du distant, puis refaire le commit et le push
ou ccréer une branche et faire un merge avec mes deux main dans la nouvelle branche je f

quand je vais le commit,

mettre le datadrift et la visualisation uniquement dans le notebook


https://redagrd-credit-scoring-streamlit-dashboard-shclty.streamlit.app/

git add .github/workflows/test.yml tests/
