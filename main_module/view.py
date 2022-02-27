# Importations

import plotly.express as px
import plotly.graph_objects as go
import plotly
import numpy as np
import pandas as pd
import wtforms.widgets
from flask import Flask, request, redirect, abort, render_template, session, flash
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *
import json

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
    # session.clear()
    return render_template('home.html')


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    _choix_dataset = None
    if request.method == "POST":
        session.clear()
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
        if request.method == "POST":
            selected_nom_col_analyse = request.form.keys()
            session['selected_nom_col_analyse'] = list(selected_nom_col_analyse)
            caract_col_analyse = column_caract(df, selected_nom_col_analyse)

        elif 'selected_nom_col_analyse' in session.keys():
            caract_col_analyse = column_caract(df, session['selected_nom_col_analyse'])

        else:
            caract_col_analyse = None

        return render_template("analyse_colonnes.html", nom_col=nom_col,
                               row_data=list(df.values.tolist()), zip=zip, caract_col_analyse=caract_col_analyse)
    else:
        return render_template("waiting_for_data.html")


@app.route('/matrice_correlation', methods=['GET', 'POST'])
def matrice_correlation():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        nom_col = col_numeric(df)
        all_col = df.columns.values
        df_sans_NaN = pd.concat([df[col] for col in all_col], axis=1).dropna()
        if request.method == "POST":
            selected_nom_col_matrice_corr = request.form.keys()
            couleur_matrice = request.form.get('couleur_matrice')

            if couleur_matrice:
                session['selected_nom_col_matrice_corr'] = list(selected_nom_col_matrice_corr)[:-1]
                session['couleur_matrice'] = str(couleur_matrice)
                print(session['couleur_matrice'])
                fig = px.scatter_matrix(df_sans_NaN,
                                        dimensions=col_numeric(df_sans_NaN[session['selected_nom_col_matrice_corr']]),
                                        color=session['couleur_matrice'])
            else:
                session['selected_nom_col_matrice_corr'] = list(selected_nom_col_matrice_corr)
                fig = px.scatter_matrix(df_sans_NaN,
                                        dimensions=col_numeric(df_sans_NaN[session['selected_nom_col_matrice_corr']]))

        elif 'selected_nom_col_matrice_corr' in session.keys():
            if 'couleur_matrice' in session.keys():
                fig = px.scatter_matrix(df_sans_NaN,
                                        dimensions=col_numeric(df_sans_NaN[session['selected_nom_col_matrice_corr']]),
                                        color=session['couleur_matrice'])
            else:
                fig = px.scatter_matrix(df_sans_NaN,
                                        dimensions=col_numeric(df_sans_NaN[session['selected_nom_col_matrice_corr']]))
        else:
            fig = None
        if fig:
            fig.update_layout(width=700, height=500, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
            fig.update_layout({"xaxis" + str(i + 1): dict(showticklabels=False) for i in
                               range(len(col_numeric(df_sans_NaN[session['selected_nom_col_matrice_corr']])))})
            fig.update_layout({"yaxis" + str(i + 1): dict(showticklabels=False) for i in
                               range(len(col_numeric(df_sans_NaN[session['selected_nom_col_matrice_corr']])))})
            fig.update_traces(marker=dict(size=4))
            fig.update_traces(diagonal_visible=False)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template("matrice_correlation.html", nom_col=nom_col, fig=fig, all_col=all_col)
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
