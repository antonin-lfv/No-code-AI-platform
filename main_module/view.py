# Importations
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import wtforms.widgets
from flask import Flask, request, redirect, abort, render_template, session, flash
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *

from main_module.static.utils import *
from main_module.static.classes import *

app = Flask(__name__)
app.config.from_object('config')
bootstrap = Bootstrap(app)

dico_dataset = {
    'iris': pd.read_csv("main_module/static/datasets/iris.csv"),
    'penguins': pd.read_csv("main_module/static/datasets/penguins.csv"),
    'voitures': pd.read_csv("main_module/static/datasets/CarPrice.csv"),
}


# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
def home():
    # pour clear les objets enregistrés sur la session dès qu'on revient au menu
    session.clear()
    return render_template('home.html')


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    _choix_dataset = None
    if request.method == "POST":
        _choix_dataset = request.form.get('dataset')
        session['_choix_dataset'] = _choix_dataset
        if session['_choix_dataset'] in dico_dataset.keys():
            df = dico_dataset[session['_choix_dataset']]
            caract_dataset = all_caract(df)
            return render_template('dataset.html', select_dataset=_choix_dataset, column_names=df.columns.values,
                                   row_data=list(df.values.tolist()), zip=zip, caract_dataset=caract_dataset)
        else:
            return render_template('dataset.html', select_dataset=_choix_dataset)
    else:
        if '_choix_dataset' in session.keys():
            df = dico_dataset[session['_choix_dataset']]
            caract_dataset = all_caract(df)
            return render_template('dataset.html', select_dataset=_choix_dataset, column_names=df.columns.values,
                                   row_data=list(df.values.tolist()), zip=zip, caract_dataset=caract_dataset)
        else:
            return render_template('dataset.html', select_dataset=_choix_dataset)


@app.route('/analyse_colonnes', methods=['GET', 'POST'])
def analyse_colonnes():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        nom_col = df.columns.values
        selected_nom_col, caract_col = None, None
        if request.method == "POST":
            selected_nom_col = request.form.keys()
            caract_col = column_caract(df, selected_nom_col)
        return render_template("analyse_colonnes.html", nom_col=nom_col, selected_nom_col=selected_nom_col,
                               row_data=list(df.values.tolist()), zip=zip, caract_col=caract_col)
    else:
        return render_template("waiting_for_data.html")


@app.route('/matrice_correlation', methods=['GET', 'POST'])
def matrice_correlation():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("matrice_correlation.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/section_graphique', methods=['GET', 'POST'])
def section_graphiques():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("section_graphiques.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/regressions', methods=['GET', 'POST'])
def regressions():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("regressions.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/KNN', methods=['GET', 'POST'])
def KNN():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("KNN.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/KMeans', methods=['GET', 'POST'])
def KMeans():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("KMeans.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/SVM', methods=['GET', 'POST'])
def SVM():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("SVM.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("decision_tree.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/PCA', methods=['GET', 'POST'])
def PCA():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("PCA.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/UMAP', methods=['GET', 'POST'])
def UMAP():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("UMAP.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/TSNE', methods=['GET', 'POST'])
def TSNE():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("TSNE.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("waiting_for_data.html")


"""
@app.route('/<string:name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

if __name__ == '__main__':
    app.run()
"""
