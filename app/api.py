import matplotlib
import matplotlib.pyplot as plt
import json
from typing import List
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Initialisation de l'application FastAPI
app = FastAPI()

# Charger les données
df = pd.read_csv(r"C:\Users\John\Desktop\Formation\7-Implémentez un modèle de scoring\data\df_.csv")[0:100]
#df.drop(columns='index', inplace=True)

# Charger les variables threshold et important features
# Opening JSON file
f = open('../model/data.json')
  
# returns JSON object as a dictionary
data = json.load(f)

# Charger le meilleur seuil
best_th = data['best_th']

# Charger la liste des features importantes
L_var = data['feat']

# Charger le meilleur modèle
best_model = pickle.load(open('../model/model.pkl', 'rb'))

# Fonction de graphe
def graphe(df, num, c_u, title):
     
    df_u = df.loc[df['TARGET']!=3,c_u+['TARGET', 'SK_ID_CURR']]

    dfg = df_u.groupby('TARGET').mean()
    dfg = dfg.drop(columns='SK_ID_CURR')

    l = pd.concat([dfg,df.loc[df['SK_ID_CURR']==num, c_u]],ignore_index=True)
    # Transformation of data for better vizualisation
    l = l.abs()+1
    l = np.log(l)

    X = l.columns

    credit_accepted = l.iloc[0,:]
    credit_refused = l.iloc[1,:]
    customer = l.iloc[2,:]

    X_axis = np.arange(l.shape[1])

    fig = plt.figure(figsize=(10,5))

    plt.bar(X_axis - 0.2, credit_accepted, 0.2, label = 'credit_accepted')
    plt.bar(X_axis + 0, credit_refused, 0.2, label = 'credit_refused')
    plt.bar(X_axis + 0.2, customer, 0.2, label = 'customer')

    plt.xticks(X_axis, X, rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.title(title)
    plt.legend()
    plt.show()

    return fig


@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API du projet 7 - Implémentez un modèle de scoring"}

@app.get("/predict/{num}")
def predict(num: int):
    
    data = df.loc[df['SK_ID_CURR']==num, L_var].values

    y_te_pred = best_model.predict(data)
    y_te_pred = (y_te_pred >= best_th)
        
    y_proba = best_model.predict_proba(data)
    proba = y_proba[0][1]

    if proba <= best_th:
        result = {
            "prediction": "Crédit accordé",
            "risque_defaut": round(proba, 2)
        }
    else:
        result = {
            "prediction": "Crédit refusé",
            "risque_defaut": round(proba, 2)
        }
    
    return result

@app.get("/graph/{num}")
def graph(num: int):
    fig = graphe(df, num, L_var, 'customer vs total population')
    return fig
    
@app.get("/graph/similar/{num}")
def graph_similar(num: int):
    sex = int(df.loc[df['SK_ID_CURR']==num, 'CODE_GENDER'])
    age = int(df.loc[df['SK_ID_CURR']==num, 'DAYS_BIRTH'])
            
    mask = (df['DAYS_BIRTH'] <= age+364) & (df['DAYS_BIRTH'] > age-364)
    df_s = df.loc[(mask==True)&(df['CODE_GENDER']==sex),:].reset_index(drop=True)
    
    fig = graphe(df_s, num, L_var, 'customer vs similar population')
    return fig

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
