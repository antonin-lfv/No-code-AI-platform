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
from sklearn.linear_model import LinearRegression, PoissonRegressor, ElasticNet, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import umap.umap_ as UMAP
from scipy.spatial import distance

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
        choix_col = col_numeric(df) + col_temporal(df)
        for accordion_choice in ['selected_abscisse_col_sect_graphiques', 'selected_ordonnee_col_sect_graphiques',
                                 'type_graphique_sect_graphiques', 'stats_graphique_sect_graphiques',
                                 'reg_graphique_sect_graphiques']:
            if accordion_choice != 'type_graphique_sect_graphiques' and accordion_choice not in session.keys():
                session[accordion_choice] = 'empty'
            else:
                if accordion_choice not in session.keys():
                    """ Par défaut on trace avec des points """
                    session[accordion_choice] = ['Points']

        if request.method == "POST":
            if request.form.getlist('selected_abscisse_col_sect_graphiques'):
                session['selected_abscisse_col_sect_graphiques'] = request.form.getlist(
                    'selected_abscisse_col_sect_graphiques')[0]

            if request.form.getlist('selected_ordonnee_col_sect_graphiques'):
                session['selected_ordonnee_col_sect_graphiques'] = request.form.getlist(
                    'selected_ordonnee_col_sect_graphiques')[0]

            if request.form.getlist('type_graphique_sect_graphiques'):
                session['type_graphique_sect_graphiques'] = request.form.getlist('type_graphique_sect_graphiques')[0]

            if request.form.getlist('stats_graphique_sect_graphiques'):
                session['stats_graphique_sect_graphiques'] = request.form.getlist('stats_graphique_sect_graphiques')
                if len(session['stats_graphique_sect_graphiques']) != 1:
                    # Si il y a au moins une valeur différente de empty
                    session['stats_graphique_sect_graphiques'] = session['stats_graphique_sect_graphiques'][:-1]

            if request.form.getlist('reg_graphique_sect_graphiques'):
                session['reg_graphique_sect_graphiques'] = request.form.getlist('reg_graphique_sect_graphiques')
                if len(session['reg_graphique_sect_graphiques']) != 1:
                    # Si il y a au moins une valeur différente de empty
                    session['reg_graphique_sect_graphiques'] = session['reg_graphique_sect_graphiques'][:-1]

        if session['selected_abscisse_col_sect_graphiques'] != 'empty' and session['selected_ordonnee_col_sect_graphiques'] != 'empty':
            df = dico_dataset[session['_choix_dataset']]
            df_sans_NaN = pd.concat([df[session['selected_abscisse_col_sect_graphiques']].reset_index(drop=True), df[session['selected_ordonnee_col_sect_graphiques']].reset_index(drop=True)], axis=1).dropna()
            # ajouter erreur si le df_sans_NaN est vide !

            if session['type_graphique_sect_graphiques'] == 'Latitude/Longitude':
                fig = go.Figure()
                fig.add_scattermapbox(
                    mode="markers",
                    lon=df_sans_NaN[session['selected_abscisse_col_sect_graphiques']],
                    lat=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']],
                    marker={'size': 10,
                            'color': 'firebrick',
                            })
                fig.update_layout(
                    margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                    mapbox={
                        'center': {'lon': -80, 'lat': 40},
                        'style': "stamen-terrain",
                        'zoom': 1})

            elif session['type_graphique_sect_graphiques'] == 'Histogramme':
                fig = px.histogram(df_sans_NaN, x=session['selected_abscisse_col_sect_graphiques'], y=session['selected_ordonnee_col_sect_graphiques'])

            else:
                # Courbe ou points
                fig = go.Figure()
                type_graphe = {
                    "Courbe": 'lines',
                    "Points": 'markers'
                }
                fig.add_scatter(x=df_sans_NaN[session['selected_abscisse_col_sect_graphiques']],
                                y=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']],
                                mode=type_graphe[session['type_graphique_sect_graphiques']], name='', showlegend=False)

                if 'Régression Linéaire' in session['reg_graphique_sect_graphiques']:
                    # regression linaire
                    X = df_sans_NaN[session['selected_abscisse_col_sect_graphiques']].values.reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']])
                    x_range = np.linspace(X.min(), X.max(), len(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']]))
                    y_range = model.predict(x_range.reshape(-1, 1))
                    fig.add_scatter(x=x_range, y=y_range, name='<b>Regression linéaire<b>', mode='lines',
                                    marker=dict(color='red'))
                    # #################
                if 'Régression polynomiale' in session['reg_graphique_sect_graphiques']:
                    # regression polynomiale
                    X = df_sans_NaN[session['selected_abscisse_col_sect_graphiques']].values.reshape(-1, 1)
                    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    poly = PolynomialFeatures(5)  # ajouter un slider pour choisir !
                    poly.fit(X)
                    X_poly = poly.transform(X)
                    x_range_poly = poly.transform(x_range)
                    model = LinearRegression(fit_intercept=False)
                    model.fit(X_poly, df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']])
                    y_poly = model.predict(x_range_poly)
                    fig.add_scatter(x=x_range.squeeze(), y=y_poly, name='<b>Polynomial Features<b>',
                                    marker=dict(color='green'))
                    # #################

                if 'Moyenne' in session['stats_graphique_sect_graphiques']:
                    # Moyenne #
                    fig.add_hline(y=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].mean(),
                                  line_dash="dot",
                                  annotation_text="moyenne : {}".format(
                                      round(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].mean(), 1)),
                                  annotation_position="bottom left",
                                  line_width=2, line=dict(color='black'),
                                  annotation=dict(font_size=10))
                    # #################
                    pass
                if 'Minimum' in session['stats_graphique_sect_graphiques']:
                    # Minimum #
                    fig.add_hline(y=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].min(),
                                  line_dash="dot",
                                  annotation_text="minimum : {}".format(
                                      round(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].min(), 1)),
                                  annotation_position="bottom left",
                                  line_width=2, line=dict(color='black'),
                                  annotation=dict(font_size=10))
                    # #################
                    pass
                if 'Maximum' in session['stats_graphique_sect_graphiques']:
                    # Maximum #
                    fig.add_hline(y=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].max(),
                                  line_dash="dot",
                                  annotation_text="maximum : {}".format(
                                      round(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].max(), 1)),
                                  annotation_position="top left",
                                  line_width=2, line=dict(color='black'),
                                  annotation=dict(font_size=10))
                    # #################

        else:
            fig = None

        if fig:
            fig.update_xaxes(title_text=str(session['selected_abscisse_col_sect_graphiques']))
            fig.update_yaxes(title_text=str(session['selected_ordonnee_col_sect_graphiques']))
            fig.update_layout(
                template='simple_white',
                font=dict(size=10),
                autosize=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            fig.update_layout(width=1000, height=500, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
            fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template("section_graphiques.html",
                               all_col=choix_col, fig=fig,
                               row_data=list(df.values.tolist()), zip=zip)
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
