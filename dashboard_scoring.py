import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import altair as alt
import requests
import json
import flask
import time

import lime
from lime import lime_tabular

import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import dill
from io import BytesIO
import base64

######################################
# Fonctions utilisées par l'API Flask
######################################
# Fonction de calcul de la probabilite de défaut de paiement d'un client
def predict_fn(x):
    return model.predict_proba(std_scaler.transform(transformer.transform(x))).astype(float)

# Fonction de calcul du voisinage d'un point : les 20 plus proches voisins
def f_knn(df,client_idx):

    # fit nearest neighbors among the selection
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(df)

    l_idx = neigh.kneighbors(X=df.loc[client_idx:client_idx], n_neighbors=20, return_distance=False).ravel()
    knn_idx = list(df.iloc[l_idx].index)
    return knn_idx

# Fonction de calcul de la moyenne et de la déviation standard des 20 voisins les plus proches d'un client pour une feature donnée
def f_cal_mean_std_knn(df,feature,client_idx):

    st.write("Client_idx :", client_idx)
    knn_idx = f_knn(df, client_idx)
    return df.loc[knn_idx,feature].mean(), df.loc[knn_idx,feature].std()

# Fonction de calcul de la moyenne et de la deviation standard d'une feature
def f_cal_mean_std(df,feature):

    return df[feature].mean(), df[feature].std()

##########################
# Fonction de l'API Flask
##########################

# Fonction de lecture des ID Clients
def get_clients():

    url       = 'http://127.0.0.1:5000/clients'
    response  = requests.get(url)
    data_json = response.json()
    df        = pd.read_json(data_json,orient='index')

    return df

# Fonction de Calcul de la prediction de la probabilité de défaut de paiement d'un client 
def get_prediction(id_client):
    'id_client : SK_ID_CURR' 

    #url = 'https://home-credit-risk.herokuapp.com/predict'
    url      = 'http://127.0.0.1:5000/predict'
    response = requests.get(url+'?id_client='+str(id_client))
    #st.markdown('**'+response.json()+'**')
    prob_default = round(float(response.json()),2)

    return prob_default

# Fonction de calcul de l'explicabilité d'un client
def get_explain_instance(idx_client):
    'idx_client : index du client'

    #url = 'https://home-credit-risk.herokuapp.com/predict'
    url      = 'http://127.0.0.1:5000/explain'
    response = requests.get(url+'?idx_client='+str(idx_client))
    if response.status_code==204:
        url      = 'http://127.0.0.1:5000/get_pdf/exp_html.pdf'    
        response = requests.get(url)
        #st.write('status code ',response.status_code)	

        return response

# Fonction de calcul de l'importance des Features basé sur les coeff du modele de Regression Logistique ici
def get_feature_importance():

    url       = 'http://127.0.0.1:5000/feature_importance'
    response  = requests.get(url)
    data_json = response.json()
    df         = pd.read_json(data_json,orient='index')

    return df

#####################
# Main
#####################

# Lecture des ID Clients
df_id = get_clients()
liste_id = df_id['SK_ID_CURR'].values

st.title('Application Scoring Credit')

# sauvegarde le l'ID de client pour eviter le recalcul si le client n'a pas changé
if 'id_client' not in st.session_state:
    st.session_state['id_client'] = ''

# Box de Selection de l'ID Client
id_client = st.sidebar.selectbox("Identifiant Client: ", liste_id, index=0)

idx = df_id[df_id['SK_ID_CURR']==id_client].index.values[0]

# CheckBox de Sélection de Données Client + Comparaison Client  et Calcul de Scoring 
scoring     = st.sidebar.checkbox('Scoring Prédiction')
data_client = st.sidebar.checkbox('Données Client')

# Selection Données Client + Comparaison Client
if scoring:
      st.write('ID Client N° ',id_client)
	
      # Importer du modele et application
      if st.session_state['id_client']!=id_client:
         prob_default = get_prediction(id_client)
         st.session_state['prob_default']=prob_default
	 
      if st.session_state['prob_default'] > 0.5:
            prevision= 'Demande Rejettée'
      else:
            prevision= 'Demande Acceptée'

      # affichage prévision
      st.subheader('Statut Demande Crédit')
      st.write(prevision)
      st.subheader('Scoring Client (%)')
      st.write('La probabilité de Défaut de paiement est de ',(st.session_state['prob_default']*100), ' %')

      # Calcul de l'Interprétabilité avec calcul de l'explication d'une instance avec Lime
      st.subheader('Explication du Score')
      if st.session_state['id_client']!=id_client:
         with st.spinner('Calcul en cours...'):
              response = get_explain_instance(idx)

              # On convertit le PDF en format base64 pour pouvoir l'afficher
              f = BytesIO(response.content)
              base64_pdf = base64.b64encode(f.read()).decode('utf-8')
              pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
              st.success('Fait !')
              st.session_state['pdf_display']=pdf_display

      # Affichage au format PDF de l'explication
      st.markdown(st.session_state['pdf_display'], unsafe_allow_html=True)

      # Calcul de Feature importance par les coefficients du modele
      st.subheader('Global Features Importance')
      df_feature_imp = get_feature_importance()

      fig, ax = plt.subplots()
      ax.bar(x=df_feature_imp['Attribute'],height=df_feature_imp['Importance'])
      ax.set_xticklabels(df_feature_imp['Attribute'].to_list(), rotation = 85)
      st.pyplot(fig)

      st.session_state['id_client']=id_client

# Selection des Données Client + Comparaison
if data_client:

 # Affichage Données
 st.subheader('Données Client')

 if not scoring:
      st.write('ID Client N° ',id_client)

 # Traitement Comparaison
 st.subheader('Variables Client')
 url = 'http://127.0.0.1:5000/var_client'
 response = requests.get(url+'?id_client='+str(id_client))
 liste_var=response.json()

 select_var=st.multiselect(label='Variables Comparatives',options=liste_var)

 if select_var:

    #st.write(select_var)

    for i in range(len(select_var)):

     # Graphique
     url = 'http://127.0.0.1:5000/comp_client'
     response = requests.get(url+'?id_client='+str(id_client)+'&var_client='+select_var[i])
     l_data = response.json()    

     value_var_client      = l_data[0]
     var_mean_other_client = l_data[1]     
     var_std_other_client  = l_data[2]     
     var_mean_simil_client = l_data[3]     
     var_std_simil_client  = l_data[4]     

     # closed_days_credit_var : 170653.0
     var=select_var[i]

     x_client = ["Client","Other Clients Mean","Other Clients Std","Clients Similaires Mean","Clients Similaires Std"]
     y_client = [value_var_client,var_mean_other_client,var_std_other_client,var_mean_simil_client,var_std_simil_client]

     source = pd.DataFrame({
         'Type Client': ["Client","Other Clients Mean","Other Clients Std","Clients Similaires Mean","Clients Similaires Std"],
         'Value/Mean/Std': y_client
     })
 
     bar_chart = alt.Chart(source).mark_bar().encode(
         x='Type Client',
         y='Value/Mean/Std'
     ).properties( title='Comparaison '+ var)
 
     st.altair_chart(bar_chart, use_container_width=True)

