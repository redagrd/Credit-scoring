from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Charger le modèle sauvegardé
model = joblib.load('lightgbm_model.pkl')

# Chemin vers le fichier CSV des données clients
data_file_path = 'final_processed_data_with_ids.csv'

# Charger les données clients si le fichier existe
if os.path.exists(data_file_path):
    data = pd.read_csv(data_file_path)
else:
    raise FileNotFoundError(f"Le fichier {data_file_path} est introuvable.")

# Nom de la colonne d'identifiant client
client_id_col = 'skidcurr'

# Liste des features du modèle, telle que sauvegardée
with open('model_features.txt', 'r') as f:
    model_features = f.read().splitlines()

# Créer une instance de l'application FastAPI
app = FastAPI()

# Modèle de requête pour FastAPI
class ClientIDRequest(BaseModel):
    client_id: int

@app.post("/predict")
def predict(request: ClientIDRequest):
    try:
        # Récupérer l'identifiant du client depuis la requête
        client_id = request.client_id

        # Vérifier si l'ID existe dans les données
        if client_id_col not in data.columns:
            raise HTTPException(status_code=500, detail=f"La colonne '{client_id_col}' est introuvable dans les données.")

        # Filtrer les données pour le client en fonction de l'identifiant
        client_data = data[data[client_id_col] == client_id]

        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client ID not found")

        # Supprimer les colonnes non utilisées par le modèle
        unnecessary_columns = ['target', client_id_col]
        client_data = client_data.drop(columns=unnecessary_columns, errors='ignore')

        # Vérifier que les features du client correspondent à celles du modèle
        missing_features = set(model_features) - set(client_data.columns)
        if missing_features:
            raise HTTPException(status_code=500, detail=f"Les features suivantes sont manquantes dans les données du client: {missing_features}")

        # Réordonner les colonnes selon l'ordre attendu par le modèle
        client_data = client_data[model_features]

        # Faire une prédiction avec le modèle
        prediction = model.predict(client_data)

        # Retourner la réponse sous forme de dictionnaire
        return {"client_id": client_id, "prediction": int(prediction[0])}

    except Exception as e:
        # Retourner une erreur HTTP 500 en cas de problème
        raise HTTPException(status_code=500, detail=str(e))





# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import os

# # Charger le modèle sauvegardé
# model = joblib.load('lightgbm_model.pkl')

# # Chemin vers le fichier CSV des données clients
# data_file_path = 'final_processed_data_with_ids.csv'

# # Charger les données clients si le fichier existe
# if os.path.exists(data_file_path):
#     data = pd.read_csv(data_file_path)
# else:
#     raise FileNotFoundError(f"Le fichier {data_file_path} est introuvable.")

# # Définir l'identifiant du client (remplace 'skidcurr' par le nom de ta colonne d'identifiant)
# client_id_col = 'skidcurr'

# # Créer une instance de l'application FastAPI
# app = FastAPI()

# # Modèle de requête pour FastAPI
# class ClientIDRequest(BaseModel):
#     client_id: int

# @app.post("/predict")
# def predict(request: ClientIDRequest):
#     try:
#         # Récupérer l'identifiant du client depuis la requête
#         client_id = request.client_id

#         # Vérifier si l'ID existe dans les données
#         if client_id_col not in data.columns:
#             raise HTTPException(status_code=500, detail=f"La colonne '{client_id_col}' est introuvable dans les données.")

#         # Filtrer les données pour le client en fonction de l'identifiant
#         client_data = data[data[client_id_col] == client_id]

#         if client_data.empty:
#             raise HTTPException(status_code=404, detail="Client ID not found")

#         # Supprimer les colonnes non utilisées par le modèle
#         unnecessary_columns = ['target']  # Ajouter ici toutes les colonnes qui ne sont pas des features
#         client_data = client_data.drop(columns=unnecessary_columns, errors='ignore')

#         # Faire une prédiction avec le modèle
#         prediction = model.predict(client_data)

#         # Retourner la réponse sous forme de dictionnaire
#         return {"client_id": client_id, "prediction": int(prediction[0])}

#     except Exception as e:
#         # Retourner une erreur HTTP 500 en cas de problème
#         raise HTTPException(status_code=500, detail=str(e))





# from fastapi import FastAPI, HTTPException
# import joblib
# import pandas as pd

# # Charger le modèle sauvegardé
# model = joblib.load('lightgbm_model.pkl')

# # Créer une instance de l'application FastAPI
# app = FastAPI()

# # Liste des vrais noms des features utilisés lors de l'entraînement
# all_features = [
#     'index', 'skidcurr', 'target', 'namecontracttype', 'codegender', 'flagowncar',
#     'flagownrealty', 'cntchildren', 'amtincometotal', 'amtcredit', 'amtannuity',
#     'amtgoodsprice', 'nametypesuite', 'nameincometype', 'nameeducationtype',
#     'namefamilystatus', 'namehousingtype', 'regionpopulationrelative', 'daysbirth',
#     'daysemployed', 'daysregistration', 'daysidpublish', 'owncarage', 'flagmobil',
#     'flagempphone', 'flagworkphone', 'flagcontmobile', 'flagphone', 'flagemail',
#     'occupationtype', 'cntfammembers', 'regionratingclient', 'regionratingclientwcity',
#     'weekdayapprprocessstart', 'hourapprprocessstart', 'regregionnotliveregion',
#     'regregionnotworkregion', 'liveregionnotworkregion', 'regcitynotlivecity',
#     'regcitynotworkcity', 'livecitynotworkcity', 'organizationtype', 'extsource1',
#     'extsource2', 'extsource3', 'apartmentsavg', 'basementareaavg',
#     'yearsbeginexpluatationavg', 'yearsbuildavg', 'commonareaavg', 'elevatorsavg',
#     'entrancesavg', 'floorsmaxavg', 'floorsminavg', 'landareaavg', 'livingapartmentsavg',
#     'livingareaavg', 'nonlivingapartmentsavg', 'nonlivingareaavg', 'apartmentsmode',
#     'basementareamode', 'yearsbeginexpluatationmode', 'yearsbuildmode', 'commonareamode',
#     'elevatorsmode', 'entrancesmode', 'floorsmaxmode', 'floorsminmode', 'landareamode',
#     'livingapartmentsmode', 'livingareamode', 'nonlivingapartmentsmode',
#     'nonlivingareamode', 'apartmentsmedi', 'basementareamedi',
#     'yearsbeginexpluatationmedi', 'yearsbuildmedi', 'commonareamedi', 'elevatorsmedi',
#     'entrancesmedi', 'floorsmaxmedi', 'floorsminmedi', 'landareamedi',
#     'livingapartmentsmedi', 'livingareamedi', 'nonlivingapartmentsmedi',
#     'nonlivingareamedi', 'fondkapremontmode', 'housetypemode', 'totalareamode',
#     'wallsmaterialmode', 'emergencystatemode', 'obs30cntsocialcircle',
#     'def30cntsocialcircle', 'obs60cntsocialcircle', 'def60cntsocialcircle',
#     'dayslastphonechange', 'flagdocument2', 'flagdocument3', 'flagdocument4',
#     'flagdocument5', 'flagdocument6', 'flagdocument7', 'flagdocument8',
#     'flagdocument9', 'flagdocument10', 'flagdocument11', 'flagdocument12',
#     'flagdocument13', 'flagdocument14', 'flagdocument15', 'flagdocument16',
#     'flagdocument17', 'flagdocument18', 'flagdocument19', 'flagdocument20',
#     'flagdocument21', 'amtreqcreditbureauhour', 'amtreqcreditbureauday',
#     'amtreqcreditbureauweek', 'amtreqcreditbureaumon', 'amtreqcreditbureauqrt',
#     'amtreqcreditbureauyear', 'daysemployedperc', 'incomecreditperc', 'incomeperperson',
#     'annuityincomeperc', 'paymentrate'
# ]


# # Endpoint de prédiction
# @app.post("/predict")
# def predict(data: dict):
#     try:
#         # Convertir les données en DataFrame
#         df = pd.DataFrame([data])

#         # Ajouter les colonnes manquantes avec des valeurs par défaut (par exemple, 0)
#         for feature in all_features:
#             if feature not in df.columns:
#                 df[feature] = 0  # Valeur par défaut

#         # Réorganiser les colonnes pour correspondre à l'ordre d'entraînement du modèle
#         df = df[all_features]

#         # Prédiction avec le modèle chargé
#         prediction = model.predict(df)

#         # Retourner la réponse sous forme de dictionnaire
#         return {"prediction": int(prediction[0])}
    
#     except Exception as e:
#         # Retourner une erreur HTTP 500 en cas de problème
#         raise HTTPException(status_code=500, detail=str(e))




# from fastapi import FastAPI
# import joblib
# import pandas as pd

# # Charger le modèle sauvegardé
# model = joblib.load('lightgbm_model.pkl')

# # Créer une instance de l'application FastAPI
# app = FastAPI()

# # Endpoint de prédiction
# @app.post("/predict")
# def predict(data: dict):
#     # Convertir les données en DataFrame
#     df = pd.DataFrame([data])

#     # Prédiction avec le modèle chargé
#     prediction = model.predict(df)

#     # Retourner la réponse sous forme de dictionnaire
#     return {"prediction": int(prediction[0])}
