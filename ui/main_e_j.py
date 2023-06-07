# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(layout="wide")
import pickle
import pandas as pd
import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import json
import requests

import shap
from shap.plots import waterfall, force


#load data
df = pd.read_csv("data/df_.csv")[0:20]
X = pd.read_csv("data/X.csv")

# Définir la première colonne en tant qu'index
X = X.set_index(X.iloc[:, 0])

# Supprimer la première colonne du DataFrame
X = X.iloc[:, 1:]

# Charger les variables threshold et important features
# Opening JSON file
f = open('model/data.json')#'../model/data.json')

# returns JSON object as a dictionary
data = json.load(f)

# Charger le meilleur seuil
best_th = data['best_th']

# Charger la liste des features importantes
L_var = data['feat']

# Charger Explainer
with open('ui/exp.pickle', 'rb') as file:
    exp = pickle.load(file)

def main():

    # Titre de la page
    st.title("Projet 7 - Implémentez un modèle de scoring")
    st.text("Données client : ")

    values = df['SK_ID_CURR'].values
    num = st.sidebar.selectbox(
        "Veuillez sélectionner un numéro de demande de prêt",
        values)
    st.sidebar.write(f"Situation familiale : {[mot[19:] for mot in df.columns if (('FAMILY' in mot)&(int(df.loc[df['SK_ID_CURR']==num,mot])))][0]}")
    st.sidebar.write(f"Nombre d'enfant(s) : {int(df.loc[df['SK_ID_CURR']==num,'CNT_CHILDREN'])}")
    st.sidebar.write(f"Age : {round(int(df.loc[df['SK_ID_CURR']==num, 'DAYS_BIRTH'])/(-364))}")    
    st.write(df.loc[df['SK_ID_CURR']==num, L_var])

    #Le bouton de prédiction
    input_data = {'SK_ID_CURR':int(num)}

    if st.button("Prediction"):
        result = requests.post(url="http://monapp.herokuapp.com/predict",data=json.dumps(input_data))
        #result = requests.post(url="http://127.0.0.1:8000/predict",data=json.dumps(input_data))
        result=result.json()
        p=result['prediction']
        st.text(f'Probabilité de défaillance (limite {best_th}): {p}')

    #Shap client
    idx = df[df['SK_ID_CURR'] == num].index.item()
    st.title("Client")
    waterfall(exp[idx])
    st.pyplot()

    #Shap global
    st.title('client global moyen')
    idx = X.index.get_loc('mean')
    waterfall(exp[idx])
    st.pyplot()

    #Shap similaire
    # fonction pour récupérer l'âge d'un client
    def roundDown(n):
        a=int(-n/3640)
        return 10*a
    st.title('client similaire - même sexe et même décennie de naissance')
    sex = int(df.loc[df['SK_ID_CURR']== num, 'CODE_GENDER'])
    age = int(df.loc[df['SK_ID_CURR']== num, 'DAYS_BIRTH'])

    index = 's' + str(sex) + 'm' + str(roundDown(age))

    idx = X.index.get_loc(index)

    waterfall(exp[idx])
    st.pyplot()

# Tests unitaires    

import subprocess

def run_tests():
    command = "pytest ./tests/test_P7.py"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Test réussi !")
        main()
    else:
        print("Test échoué.")
        print(result.stdout)
        st.title(f"Les tests ont échoué - code : {result.returncode}")

if __name__ == '__main__':
    run_tests()
