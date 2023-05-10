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

   #chargement des données Valid/Test
df_i=pd.read_csv("../data/df_i.csv")

def main():

    # Fonction de prédiction
    @st.cache_data
    def predict (data):
        best_model = pickle.load(open('../model/Best_predictor.pkl', 'rb'))
        y_te_pred = best_model.predict(data)
        y_te_pred = (y_te_pred >= 0.52)
        
        y_proba = best_model.predict_proba(data)

        return y_te_pred, y_proba

    # Vizualisation
    @st.cache_data
    def graphe(num):

        c_u = df_i.drop(columns=['TARGET','SK_ID_CURR']).columns.tolist()

        dfg = df_i.groupby('TARGET').mean()
        dfg = dfg.drop(columns='SK_ID_CURR')

        l = pd.concat([dfg,df_i.loc[df_i['SK_ID_CURR']==num, c_u]],ignore_index=True)

        # Vizualisation
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
        #plt.xticks(rotation=45,fontsize=12)
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.title("Values per feature")
        plt.legend()
        plt.show()

        return fig

    
    #Titre de la page
    st.title("Projet 7 - Implémentez un modèle de scoring")

    #Les box de remplissage des données

    # Variable list
    L_var = df_i.columns.drop(['TARGET','SK_ID_CURR'])
    L_var = L_var.tolist()

    st.title("Credit Predictor")

    values = df_i['SK_ID_CURR'].values
    num = st.sidebar.selectbox(
        "Veuillez sélectionner un numéro de demande de prêt",
        values)

    input_data = {}
    for i in L_var :
        input_data[i] = st.number_input(i, value=df_i.loc[df_i['SK_ID_CURR']==num, i].tolist()[0])
    
    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(df_i.loc[df_i['SK_ID_CURR']==num,:])
    
    #Le bouton de prédiction

    if st.button("Prediction"):
        
        result = predict(np.array([[input_data[i] for i in L_var]]))[0]
        proba = predict(np.array([[input_data[i] for i in L_var]]))[1][0][1]
        
        if result[0]==False:
            st.text('Crédit accordé')
            st.text(f'Proba de défaillance (limite 0.52): {round(proba,2)}')
            
        else : 
            st.text('Crédit refusé')
            st.text(f'risque de défaut de crédit (limite 52%): {round(proba,2)}')
            
    # Appeler la fonction graphe() à l'intérieur de st.pyplot()
    fig = graphe(num)
    st.pyplot(fig)
            
   
if __name__ == '__main__':
    main()

