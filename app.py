# Creat eAPI of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from github import Github
from flask import Flask, request, jsonify, Response
from flask import make_response, render_template
from flask import render_template_string
import pickle
import pandas as pd
import json
import dill
import pdfkit
from pdfrw import PdfReader
from fpdf import FPDF
from flask import Flask,send_file,send_from_directory
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

#app.config["CLIENT_PDF"] = "./pdf_files"
#app.config["CLIENT_PDF"] = "./pdf_files"

app.config["CLIENT_PDF"] = "/tmp"

############################################################
# Chargement des Data + Model + Explainer + Encoder One-Hot 
############################################################
# Load client 
def load_client():

    df_client = pd.read_csv("./idclient.csv",index_col=0)

    return df_client.iloc[0:1000]

# Load du Dataframe Data Train
def load_train_data():

    df_train = pd.read_pickle("./f_train.pkl")
    return df_train

# Load du Best Model GridSearchCV utilisé avec Lime
def load_model():

    file_model=open('./f_gil_lr.pkl','rb')
    model = pickle.load(file_model)
    file_model.close()

    return model

# Load du Scaler Lime
def load_std_scaler():

    file_scaler=open('./f_std_scaler_lime.dat','rb')
    std_scaler=pickle.load(file_scaler)
    file_scaler.close()

    return std_scaler 

# Load Transformer One-Hot Encoder
def load_transformer():

    file_transformer=open('./f_transformer.pkl','rb')
    transformer=pickle.load(file_transformer)
    file_transformer.close()

    return transformer

# Load Explainer Lime
def load_explainer():

    file_explain=open('./f_explainer.dat','rb')
    explainer=dill.load(file_explain)
    file_explain.close()

    return explainer

# Load Feature Importance Logistic Regression
def load_feature_imp():

    file_feat_imp=open('./f_feature_imp_lr.dat','rb')
    feat_imp=pickle.load(file_feat_imp)

    return feat_imp

##############################################################
# Fonctions de calcul utlisés pour la comparaison des clients
##############################################################
# La comparaison des clients se fait sur :
#  -  1 seule dimension Feature donnéen
#  -  en comparant les moyennes et deviation standard de tous les client  et du client donné
#  -  en comparant les moyennes et deviation standard de clients similaires et du client donné

# Fonction de calcul du voisinage d'un point : les 20 plus proches voisin
def f_knn(df,client_idx):

    # fit nearest neighbors among the selection
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(df)

    l_idx = neigh.kneighbors(X=df.loc[client_idx:client_idx], n_neighbors=20, return_distance=False).ravel()
    knn_idx = list(df.iloc[l_idx].index)
    return knn_idx

# Fonction de calcul de la moyenne et de la déviation standard des 20 voisins les plus proches d'un client pour une feature donnée
def f_cal_mean_std_knn(df,feature,client_idx):

    print("Client_idx :", client_idx)
    knn_idx = f_knn(df, client_idx)
    return df.loc[knn_idx,feature].mean(), df.loc[knn_idx,feature].std()

# Fonction de calcul de la moyenne et de la déviation standard d'une feature
def f_cal_mean_std(df,feature):

    return df[feature].mean(), df[feature].std() 


#####################################
# Routes et Fonctiosn de l'API Flask
#####################################

@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"text":"Hello, the API is up and running..." })

# Fonction de lecture des ID Clients
@app.route("/clients")
def clients():

    df = load_client()
    data_json=df.to_json(orient='index')

    return jsonify(data_json)


# Fonction de Calcul de la probabilité de défaut de paiement d'un client id_client
@app.route('/predict', methods=['GET'])
def predict():

    std_scaler  = load_std_scaler()
    transformer = load_transformer()
    model       = load_model()

    df_train = load_train_data()
    df_client = load_client()

    args = request.args
    #print('args = ',args)
    #print('args[id_client] = ',args['id_client'])
    id_client=int(args['id_client'])
    print('id_client = ', id_client)

    idx = df_client[df_client['SK_ID_CURR']==id_client].index
    print('idx = ',idx)
    X   = df_train.loc[idx]
    print('X = ',X)

    # Standardisation des données
    X_ts = np.array(std_scaler.transform(transformer.transform(X)))

    # Calcul de la probabilité de défaut
    prob_default = model.best_estimator_.predict_proba(X_ts)[:, 1][0]
    prediction = "Le client a " + str(round(prob_default*100,2)) + "% de risque de défault de paiement"
    print("prediction: ", prediction)

    return jsonify(json.dumps(prob_default))

# Fonction de Prédiction utilisée par la fonction predict() ci-dessus
def predict_fn(x):

    std_scaler  = load_std_scaler()
    transformer = load_transformer()
    model       = load_model()

    return model.predict_proba(std_scaler.transform(transformer.transform(x))).astype(float)

# Fonction de Calcul de la Feature Importance
@app.route('/feature_importance', methods=['GET'])
def feature_importance():
     
    df_feat_imp = load_feature_imp()
    data_json=df_feat_imp.to_json(orient='index')

    return jsonify(data_json)


# Fonction de calcul de l'explicabilité d'un client
@app.route("/explain")
def explain():
    """
    explain an instance with Lime 
    """
    df_train = load_train_data()
    explainer = load_explainer()

    print('########################')
    print('request.args = ', request.args)
    args = request.args
    print('args = ',args)
   
    idx_client=args.get('idx_client')
    idx_client=int(idx_client)
    
    print('########################')
    print('1. idx client =', idx_client)

    exp      = explainer.explain_instance(df_train.loc[[idx_client]].values[0],predict_fn,num_features=20,top_labels=1)
    exp_html = exp.as_html()

    #github = Github('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')
    #repository = github.get_user().get_repo('P7')
    # path in the repository
    #filename = 'exp_html.html'
    #content = exp_html 
    # create with commit message
    #f = repository.create_file(filename, "create_file via PyGithub", content)

    try:
      with open("/tmp/exp_html.html", "w") as fo:
           fo.write(exp_html)
    except IOError:
        print("Impossible d'ouvrir le ficher exp_html.html")
	abort(404)

    ###with open("exp_html.html", "r") as fr:
    ###     exp_html_str = fr.read()
    ###     fr.close()

    #pdfkit.from_url('https://www.google.co.in/','shaurya.pdf')

    # sur heroku on ne cree pas de repertoire pdf_files 
    #pdfkit.from_file('exp_html.html','./pdf_files/exp_html.pdf')
    pdfkit.from_file('/tmp/exp_html.html','/tmp/exp_html.pdf')

    #pdf=PdfReader('exp_html.pdf')

    #headers = {'Content-Type': 'text/html'}
    #return make_response(render_template_string(exp_html.html'),200, headers)

    #response = make_response(pdf.output(dest='S').encode('latin-1'))
    #response = make_response(pdf)
    #response.headers.set('Content-Disposition', 'attachment', filename='exp_html_sav.pdf')
    #response.headers.set('Content-Type', 'application/pdf')

    #print('##################')	
    #print('pdf = ',pdf)
    #print('##################')	
    #return response
    #return Response(pdf, mimetype='application/pdf', headers={"Content-Disposition": "attachment;filename=exp_html_sav.pdf" })

    #rendered = render_template( 'exp_html.html')

    #pdf = pdfkit.from_string(rendered, False)
    #pdf = pdfkit.from_string(exp_html_str, False)
    #print('##################')	
    #print('exp_html_str = ',exp_html_str)
    #print('pdf = ',pdf)
    #print('##################')	
    
    #response = make_response(pdf.output(dest='S').encode('latin-1'))
    #response.headers.set('Content-Disposition', 'attachment', filename='exp_html_sav.pdf')
    #response.headers.set('Content-Type', 'application/pdf')
    return ('', 204)

@app.route('/get_pdf/<pdf_filename>',methods = ['GET','POST'])
def get_pdf(pdf_filename):

    try:
        print('app.config["CLIENT_PDF"] = ',app.config["CLIENT_PDF"])
        return send_from_directory(directory=app.config["CLIENT_PDF"], path=pdf_filename, as_attachment=True)
    except FileNotFoundError:	
        abort(404)

@app.route('/var_client', methods=['GET'])
def var_client():

    df_train = load_train_data()

    args = request.args
    print('args = ',args)
    print('args[id_client] = ',args['id_client'])
    id_client=int(args['id_client'])
    print('id_client = ', id_client)
    
    l_cols=df_train.columns.to_list()

    return jsonify(l_cols)


@app.route('/comp_client', methods=['GET'])
def comp_client():

    df_train = load_train_data()
    args = request.args
    print('args = ',args)
    print('args[id_client] = ',args['id_client'])
    id_client=int(args['id_client'])
    print('id_client = ', id_client)
    var_client=args['var_client']
    print('var_client = ', var_client)

    idx = df_train.index.values[0]
    var_mean_other_client, var_std_other_client = f_cal_mean_std(df_train,var_client)
    var_mean_simil_client, var_std_simil_client = f_cal_mean_std_knn(df_train, var_client, idx)

    #value_var_client=df_train.loc[idx,var_client].values[0].astype(float)
    value_var_client=df_train.loc[idx,var_client].astype(float)

    l_values=[value_var_client, var_mean_other_client, var_std_other_client, var_mean_simil_client, var_std_simil_client]
    return jsonify(l_values)

#@app.route('/download')
#def download ():
#    path = "./pdf_files/exp_html.pdf"
#    return send_file(path, as_attachment=True)

#@app.route('/pdf')
#def pdf():
#    # generate some file name
#    # save the file in the `database_reports` folder used below
#    name='./pdf_files/exp_html.pdf'
#    return render_template('pdf.html', filename=name)


#@app.route('/topdf/<name>')
#def topdf(name):

#    pdf = FPDF()
#    pdf.add_page()
#    pdf.set_font("Arial", size=12)
#    pdf.cell(200, 10, txt="Welcome to Python!", ln=1, align="C")
#    pdf.output("simple_demo.pdf")

#    response = make_response(pdf.output(dest='S').encode('latin-1'))
#    response.headers.set('Content-Disposition', 'attachment', filename=name + '.pdf')
#    response.headers.set('Content-Type', 'application/pdf')
#    return response

if __name__ == '__main__':
    app.run(debug=True)
