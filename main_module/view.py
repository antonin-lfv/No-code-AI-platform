# Importations

import plotly.express as px
import plotly.graph_objects as go
import plotly
import numpy as np
import pandas as pd
import umap
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
    session.clear()
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
        if len(df_sans_NaN) > 0:
            if request.method == "POST":
                selected_nom_col_matrice_corr = request.form.keys()
                couleur_matrice = request.form.get('couleur_matrice')

                if couleur_matrice:
                    session['selected_nom_col_matrice_corr'] = list(selected_nom_col_matrice_corr)[:-1]
                    session['couleur_matrice'] = str(couleur_matrice)
                    fig = px.scatter_matrix(df_sans_NaN,
                                            dimensions=col_numeric(
                                                df_sans_NaN[session['selected_nom_col_matrice_corr']]),
                                            color=session['couleur_matrice'])
                else:
                    session['selected_nom_col_matrice_corr'] = list(selected_nom_col_matrice_corr)
                    fig = px.scatter_matrix(df_sans_NaN,
                                            dimensions=col_numeric(
                                                df_sans_NaN[session['selected_nom_col_matrice_corr']]))

            elif 'selected_nom_col_matrice_corr' in session.keys():
                if 'couleur_matrice' in session.keys():
                    fig = px.scatter_matrix(df_sans_NaN,
                                            dimensions=col_numeric(
                                                df_sans_NaN[session['selected_nom_col_matrice_corr']]),
                                            color=session['couleur_matrice'])
                else:
                    fig = px.scatter_matrix(df_sans_NaN,
                                            dimensions=col_numeric(
                                                df_sans_NaN[session['selected_nom_col_matrice_corr']]))
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

            return render_template("matrice_correlation.html", nom_col=nom_col, fig=fig, all_col=all_col, erreur=None)
        else:
            return render_template("matrice_correlation.html", nom_col=nom_col, all_col=all_col, erreur=True)
    else:
        return render_template("waiting_for_data.html")


@app.route('/section_graphique', methods=['GET', 'POST'])
def section_graphiques():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        choix_col = col_numeric(df) + col_temporal(df)
        fig, erreur = None, None
        for accordion_choice in ['selected_abscisse_col_sect_graphiques', 'selected_ordonnee_col_sect_graphiques',
                                 'type_graphique_sect_graphiques', 'stats_graphique_sect_graphiques',
                                 'reg_graphique_sect_graphiques']:
            if accordion_choice != 'type_graphique_sect_graphiques' and accordion_choice not in session.keys():
                session[accordion_choice] = 'empty'
            else:
                if accordion_choice not in session.keys():
                    """ Par défaut on trace avec des points """
                    session[accordion_choice] = 'Points'

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

        if session['selected_abscisse_col_sect_graphiques'] != 'empty' and session[
            'selected_ordonnee_col_sect_graphiques'] != 'empty':
            df = dico_dataset[session['_choix_dataset']]
            df_sans_NaN = pd.concat([df[session['selected_abscisse_col_sect_graphiques']].reset_index(drop=True),
                                     df[session['selected_ordonnee_col_sect_graphiques']].reset_index(drop=True)],
                                    axis=1).dropna()
            if len(df_sans_NaN) > 0:
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
                    fig = px.histogram(df_sans_NaN, x=session['selected_abscisse_col_sect_graphiques'],
                                       y=session['selected_ordonnee_col_sect_graphiques'])

                else:
                    # Courbe ou points
                    fig = go.Figure()
                    type_graphe = {
                        "Courbe": 'lines',
                        "Points": 'markers'
                    }
                    fig.add_scatter(x=df_sans_NaN[session['selected_abscisse_col_sect_graphiques']],
                                    y=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']],
                                    mode=type_graphe[session['type_graphique_sect_graphiques']], name='',
                                    showlegend=False)

                    if 'Régression Linéaire' in session['reg_graphique_sect_graphiques']:
                        # regression linaire
                        X = df_sans_NaN[session['selected_abscisse_col_sect_graphiques']].values.reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X, df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']])
                        x_range = np.linspace(X.min(), X.max(),
                                              len(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']]))
                        y_range = model.predict(x_range.reshape(-1, 1))
                        fig.add_scatter(x=x_range, y=y_range, name='Regression linéaire', mode='lines',
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
                        fig.add_scatter(x=x_range.squeeze(), y=y_poly, name='Polynomial Features',
                                        marker=dict(color='green'))
                        # #################

                    if 'Moyenne' in session['stats_graphique_sect_graphiques']:
                        # Moyenne #
                        fig.add_hline(y=df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].mean(),
                                      line_dash="dot",
                                      annotation_text="moyenne : {}".format(
                                          round(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].mean(),
                                                1)),
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
                                          round(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].min(),
                                                1)),
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
                                          round(df_sans_NaN[session['selected_ordonnee_col_sect_graphiques']].max(),
                                                1)),
                                      annotation_position="top left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
            else:
                erreur = True

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
            fig.update_layout(width=1000, height=500, font=dict(size=10))
            fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template("section_graphiques.html",
                               all_col=choix_col, fig=fig, erreur=erreur)
    else:
        return render_template("waiting_for_data.html")


@app.route('/regressions', methods=['GET', 'POST'])
def regressions():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        choix_col = col_numeric(df)
        fig, erreur = None, None
        figures = []
        if request.method == "POST":
            if request.form.getlist('selected_features_regressions'):
                session['selected_features_regressions'] = request.form.getlist('selected_features_regressions')
            if request.form.getlist('selected_target_regressions'):
                session['selected_target_regressions'] = request.form.getlist('selected_target_regressions')[0]

            if 'selected_target_regressions' in session and 'selected_features_regressions' in session:
                if session['selected_target_regressions'] in session['selected_features_regressions'] and session['selected_target_regressions'] is not None:
                    erreur = True

        if 'selected_features_regressions' in session and 'selected_target_regressions' in session:
            if not erreur:
                # # On a tout ce qu'il faut
                df_sans_NaN = pd.concat([df[col].reset_index(drop=True) for col in [
                    session['selected_features_regressions'] + [session['selected_target_regressions']]]],
                                        axis=1).dropna()
                if len(df_sans_NaN) > 0:
                    # # Début des régressions
                    try:
                        # # On applique les modèles si il n'y a pas d'erreurs

                        # Data pour tous les modèles
                        X_train, X_test, y_train, y_test = train_test_split(
                            df_sans_NaN[session['selected_features_regressions']].values,
                            df_sans_NaN[session['selected_target_regressions']], test_size=0.4, random_state=4)
                        X_train, X_test, y_train, y_test = scale(X_train), scale(X_test), scale(y_train), scale(y_test)

                        # ###############################################################################

                        # Modèle régression linéaire
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        pred_train = model.predict(X_train)
                        pred_test = model.predict(X_test)
                        # Métrique train set
                        MSE_reg_train = mean_squared_error(y_train, pred_train)
                        RMSE_reg_train = np.sqrt(MSE_reg_train)
                        MAE_reg_train = mean_absolute_error(y_train, pred_train)
                        r2_reg_train = r2_score(y_train, pred_train)
                        # Métrique test set
                        MSE_reg_test = mean_squared_error(y_test, pred_test)
                        RMSE_reg_test = np.sqrt(MSE_reg_test)
                        MAE_reg_test = mean_absolute_error(y_test, pred_test)
                        r2_reg_test = r2_score(y_test, pred_test)
                        # Affichage métriques
                        session["reg_lineaire"] = []
                        session["reg_lineaire"].append([round(MSE_reg_test, 3), round(MSE_reg_test - MSE_reg_train, 3)])
                        session["reg_lineaire"].append(
                            [round(RMSE_reg_test, 3), round(RMSE_reg_test - RMSE_reg_train, 3)])
                        session["reg_lineaire"].append([round(MAE_reg_test, 3), round(MAE_reg_test - MAE_reg_train, 3)])
                        session["reg_lineaire"].append([round(r2_reg_test, 3), round(r2_reg_test - r2_reg_train, 3)])
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                        fig = go.Figure()
                        fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train', marker=dict(color='royalblue'))
                        fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='coral'))
                        fig.update_xaxes(title_text="Données de validation")
                        fig.update_yaxes(title_text="Score")
                        fig.update_layout(
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title={'text': "<b>Learning curves</b>",
                                   'y': 0.9,
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'
                                   }
                        )
                        figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

                        # ###############################################################################

                        # Modèle régression polynomiale
                        model1 = PolynomialFeatures(degree=4)
                        x_poly = model1.fit_transform(X_train)
                        model2 = LinearRegression(fit_intercept=False)
                        model2.fit(x_poly, y_train)
                        y_poly_pred_train = model2.predict(x_poly)
                        y_poly_pred_test = model2.predict(model1.fit_transform(X_test))
                        # Métrique train set
                        MSE_reg_train = mean_squared_error(y_train, y_poly_pred_train)
                        RMSE_reg_train = np.sqrt(MSE_reg_train)
                        MAE_reg_train = mean_absolute_error(y_train, y_poly_pred_train)
                        r2_reg_train = r2_score(y_train, y_poly_pred_train)
                        # Métrique test set
                        MSE_reg_test = mean_squared_error(y_test, y_poly_pred_test)
                        RMSE_reg_test = np.sqrt(MSE_reg_test)
                        MAE_reg_test = mean_absolute_error(y_test, y_poly_pred_test)
                        r2_reg_test = r2_score(y_test, y_poly_pred_test)
                        # Affichage métriques
                        session["reg_poly"] = []
                        session["reg_poly"].append([round(MSE_reg_test, 3), round(MSE_reg_test - MSE_reg_train, 3)])
                        session["reg_poly"].append([round(RMSE_reg_test, 3), round(RMSE_reg_test - RMSE_reg_train, 3)])
                        session["reg_poly"].append([round(MAE_reg_test, 3), round(MAE_reg_test - MAE_reg_train, 3)])
                        session["reg_poly"].append([round(r2_reg_test, 3), round(r2_reg_test - r2_reg_train, 3)])
                        # Learning curves
                        N, train_score, val_score = learning_curve(model2, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                        fig = go.Figure()
                        fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train', marker=dict(color='royalblue'))
                        fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='coral'))
                        fig.update_xaxes(title_text="Données de validation")
                        fig.update_yaxes(title_text="Score")
                        fig.update_layout(
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title={'text': "<b>Learning curves</b>",
                                   'y': 0.9,
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'
                                   }
                        )
                        figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

                        # ###############################################################################

                        if np.issubdtype(y_train.dtype, int) and np.any(y_train < 0):
                            # Modèle régression de poisson
                            model = PoissonRegressor()
                            model.fit(X_train, y_train)
                            pred_train = model.predict(X_train)
                            pred_test = model.predict(X_test)
                            # Métrique train set
                            MSE_reg_train = mean_squared_error(y_train, pred_train)
                            RMSE_reg_train = np.sqrt(MSE_reg_train)
                            MAE_reg_train = mean_absolute_error(y_train, pred_train)
                            r2_reg_train = r2_score(y_train, pred_train)
                            # Métrique test set
                            MSE_reg_test = mean_squared_error(y_test, pred_test)
                            RMSE_reg_test = np.sqrt(MSE_reg_test)
                            MAE_reg_test = mean_absolute_error(y_test, pred_test)
                            r2_reg_test = r2_score(y_test, pred_test)
                            # Affichage métriques
                            session["reg_poisson"] = []
                            session["reg_poisson"].append(
                                [round(MSE_reg_test, 3), round(MSE_reg_test - MSE_reg_train, 3)])
                            session["reg_poisson"].append(
                                [round(RMSE_reg_test, 3), round(RMSE_reg_test - RMSE_reg_train, 3)])
                            session["reg_poisson"].append(
                                [round(MAE_reg_test, 3), round(MAE_reg_test - MAE_reg_train, 3)])
                            session["reg_poisson"].append([round(r2_reg_test, 3), round(r2_reg_test - r2_reg_train, 3)])
                            # Learning curves
                            N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                       train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                            fig = go.Figure()
                            fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train',
                                            marker=dict(color='royalblue'))
                            fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation',
                                            marker=dict(color='coral'))
                            fig.update_xaxes(title_text="Données de validation")
                            fig.update_yaxes(title_text="Score")
                            fig.update_layout(
                                template='simple_white',
                                font=dict(size=10),
                                autosize=False,
                                width=900, height=450,
                                margin=dict(l=40, r=40, b=40, t=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                title={'text': "<b>Learning curves</b>",
                                       'y': 0.9,
                                       'x': 0.5,
                                       'xanchor': 'center',
                                       'yanchor': 'top'
                                       }
                            )
                            figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                        else:
                            # Régression de poisson impossible
                            figures.append('erreur')

                        # ###############################################################################

                        # Modèle Elastic net
                        model = ElasticNet()
                        model.fit(X_train, y_train)
                        pred_train = model.predict(X_train)
                        pred_test = model.predict(X_test)
                        # Métrique train set
                        MSE_reg_train = mean_squared_error(y_train, pred_train)
                        RMSE_reg_train = np.sqrt(MSE_reg_train)
                        MAE_reg_train = mean_absolute_error(y_train, pred_train)
                        r2_reg_train = r2_score(y_train, pred_train)
                        # Métrique test set
                        MSE_reg_test = mean_squared_error(y_test, pred_test)
                        RMSE_reg_test = np.sqrt(MSE_reg_test)
                        MAE_reg_test = mean_absolute_error(y_test, pred_test)
                        r2_reg_test = r2_score(y_test, pred_test)
                        # Affichage métriques
                        session["reg_elastic_net"] = []
                        session["reg_elastic_net"].append(
                            [round(MSE_reg_test, 3), round(MSE_reg_test - MSE_reg_train, 3)])
                        session["reg_elastic_net"].append(
                            [round(RMSE_reg_test, 3), round(RMSE_reg_test - RMSE_reg_train, 3)])
                        session["reg_elastic_net"].append(
                            [round(MAE_reg_test, 3), round(MAE_reg_test - MAE_reg_train, 3)])
                        session["reg_elastic_net"].append([round(r2_reg_test, 3), round(r2_reg_test - r2_reg_train, 3)])
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                        fig = go.Figure()
                        fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train', marker=dict(color='royalblue'))
                        fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='coral'))
                        fig.update_xaxes(title_text="Données de validation")
                        fig.update_yaxes(title_text="Score")
                        fig.update_layout(
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title={'text': "<b>Learning curves</b>",
                                   'y': 0.9,
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'
                                   }
                        )
                        figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

                        # ###############################################################################

                        # Modèle Ridge
                        model = Ridge()
                        model.fit(X_train, y_train)
                        pred_train = model.predict(X_train)
                        pred_test = model.predict(X_test)
                        # Métrique train set
                        MSE_reg_train = mean_squared_error(y_train, pred_train)
                        RMSE_reg_train = np.sqrt(MSE_reg_train)
                        MAE_reg_train = mean_absolute_error(y_train, pred_train)
                        r2_reg_train = r2_score(y_train, pred_train)
                        # Métrique test set
                        MSE_reg_test = mean_squared_error(y_test, pred_test)
                        RMSE_reg_test = np.sqrt(MSE_reg_test)
                        MAE_reg_test = mean_absolute_error(y_test, pred_test)
                        r2_reg_test = r2_score(y_test, pred_test)
                        # Affichage métriques
                        session["reg_ridge"] = []
                        session["reg_ridge"].append([round(MSE_reg_test, 3), round(MSE_reg_test - MSE_reg_train, 3)])
                        session["reg_ridge"].append([round(RMSE_reg_test, 3), round(RMSE_reg_test - RMSE_reg_train, 3)])
                        session["reg_ridge"].append([round(MAE_reg_test, 3), round(MAE_reg_test - MAE_reg_train, 3)])
                        session["reg_ridge"].append([round(r2_reg_test, 3), round(r2_reg_test - r2_reg_train, 3)])
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                        fig = go.Figure()
                        fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train', marker=dict(color='royalblue'))
                        fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='coral'))
                        fig.update_xaxes(title_text="Données de validation")
                        fig.update_yaxes(title_text="Score")
                        fig.update_layout(
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title={'text': "<b>Learning curves</b>",
                                   'y': 0.9,
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'
                                   }
                        )
                        figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

                        # ###############################################################################

                        # Modèle lasso
                        model = Lasso()
                        model.fit(X_train, y_train)
                        pred_train = model.predict(X_train)
                        pred_test = model.predict(X_test)
                        # Métrique train set
                        MSE_reg_train = mean_squared_error(y_train, pred_train)
                        RMSE_reg_train = np.sqrt(MSE_reg_train)
                        MAE_reg_train = mean_absolute_error(y_train, pred_train)
                        r2_reg_train = r2_score(y_train, pred_train)
                        # Métrique test set
                        MSE_reg_test = mean_squared_error(y_test, pred_test)
                        RMSE_reg_test = np.sqrt(MSE_reg_test)
                        MAE_reg_test = mean_absolute_error(y_test, pred_test)
                        r2_reg_test = r2_score(y_test, pred_test)
                        # Affichage métriques
                        session["reg_lasso"] = []
                        session["reg_lasso"].append([round(MSE_reg_test, 3), round(MSE_reg_test - MSE_reg_train, 3)])
                        session["reg_lasso"].append([round(RMSE_reg_test, 3), round(RMSE_reg_test - RMSE_reg_train, 3)])
                        session["reg_lasso"].append([round(MAE_reg_test, 3), round(MAE_reg_test - MAE_reg_train, 3)])
                        session["reg_lasso"].append([round(r2_reg_test, 3), round(r2_reg_test - r2_reg_train, 3)])
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                        fig = go.Figure()
                        fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train', marker=dict(color='royalblue'))
                        fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='coral'))
                        fig.update_xaxes(title_text="Données de validation")
                        fig.update_yaxes(title_text="Score")
                        fig.update_layout(
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title={'text': "<b>Learning curves</b>",
                                   'y': 0.9,
                                   'x': 0.5,
                                   'xanchor': 'center',
                                   'yanchor': 'top'
                                   }
                        )
                        figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    except:
                        # Régressions impossibles
                        erreur = True

                else:
                    # Dataset vide
                    erreur = True

        return render_template("regressions.html", choix_col=choix_col, erreur=erreur, zip=zip, figures=figures)

    else:
        return render_template("waiting_for_data.html")


@app.route('/KNN', methods=['GET', 'POST'])
def KNN():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        choix_col = df.columns.values
        erreur, message_encode, meilleur_k_knn, message_ROC = None, [], None, None
        figure_roc_knn, figure_learning_curves_knn = None, None
        message_prediction_knn = ""
        if 'selected_col_knn' not in session:
            session['selected_col_knn'] = []
        if 'col_numeriques_knn_avec_encode' not in session:
            session["col_numeriques_knn_avec_encode"] = []
        if 'selected_target_knn' not in session:
            session["selected_target_knn"] = []
        if 'for_prediction' not in session:
            session['for_prediction'] = []
        if 'predictions' not in session:
            session['predictions'] = []

        if request.method == "POST":
            if request.form.getlist('selected_col_knn'):
                session['selected_col_knn'] = request.form.getlist('selected_col_knn')
                session['selected_col_encode_knn'], session["col_numeriques_knn_avec_encode"], session['selected_target_knn'] = [], [], []
                session["col_numeriques_knn_avec_encode"] = col_numeric(df[session['selected_col_knn']])
            if request.form.getlist('selected_col_encode_knn'):
                session['selected_col_encode_knn'] = request.form.getlist('selected_col_encode_knn')[:-1]
                session["col_numeriques_knn_avec_encode"] = list(set(col_numeric(df[session['selected_col_knn']]) + session['selected_col_encode_knn']))
                session['selected_target_knn'] = []
            if request.form.getlist('selected_target_knn'):
                session['selected_target_knn'] = request.form.getlist('selected_target_knn')
                temp = session['selected_col_knn'].copy()
                temp.remove(session['selected_target_knn'][0])
                session['for_prediction'] = temp
            if request.form.getlist('predictions'):
                session['predictions'] = request.form.getlist('predictions')

        if len(session['selected_col_knn']) > 1:
            df_ml = df[session['selected_col_knn']]
            df_ml = df_ml.dropna(axis=0)
            df_origine = df_ml.copy()
            if len(df_ml) > 0:
                if session['selected_col_encode_knn']:
                    # Encodage
                    for col in session['selected_col_encode_knn']:
                        message_encode.append("Colonne " + col + "  :  " + str(df_ml[col].unique().tolist()) + " -> " + str(np.arange(len(df_ml[col].unique()))))
                        df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())), inplace=True)
                if session['selected_target_knn']:
                    try:
                        # KNN
                        y_knn = df_ml[session['selected_target_knn'][0]]  # target
                        X_knn = df_ml.drop(session['selected_target_knn'], axis=1)  # features
                        X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.4, random_state=4)
                        # Gridsearchcv
                        params = {'n_neighbors': np.arange(1, 20)}
                        grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=4)
                        grid.fit(X_train.values, y_train.values.ravel())
                        best_k = grid.best_params_['n_neighbors']
                        best_model_knn = grid.best_estimator_
                        best_model_knn.fit(X_knn.values, y_knn.values.ravel())  # on entraine le modèle

                        # Meilleurs hyper params
                        meilleur_k_knn = f'Après un GridSearchCV on prendra k = {best_k} voisins'

                        # Évaluation du modèle
                        y_pred_test = best_model_knn.predict(X_test.values)
                        y_pred_train = best_model_knn.predict(X_train.values)
                        if len(y_knn.unique()) > 2:
                            # average='micro' car nos label contiennent plus de 2 classes
                            # Test set
                            precis_test = precision_score(y_test, y_pred_test, average='micro')
                            rappel_test = recall_score(y_test, y_pred_test, average='micro')
                            F1_test = f1_score(y_test, y_pred_test, average='micro')
                            accur_test = accuracy_score(y_test, y_pred_test)
                            # Train set
                            precis_train = precision_score(y_train, y_pred_train, average='micro')
                            rappel_train = recall_score(y_train, y_pred_train, average='micro')
                            F1_train = f1_score(y_train, y_pred_train, average='micro')
                            accur_train = accuracy_score(y_train, y_pred_train)
                            session["knn_metrics"] = []
                            session["knn_metrics"].append([round(precis_test, 3), round(precis_test - precis_train, 3)])
                            session["knn_metrics"].append([round(rappel_test, 3), round(rappel_test - rappel_train, 3)])
                            session["knn_metrics"].append([round(F1_test, 3), round(F1_test - F1_train, 3)])
                            session["knn_metrics"].append([round(accur_test, 3), round(accur_test - accur_train, 3)])

                        else:
                            # label binaire
                            # Test set
                            precis_test = precision_score(y_test, y_pred_test)
                            rappel_test = recall_score(y_test, y_pred_test)
                            F1_test = f1_score(y_test, y_pred_test)
                            accur_test = accuracy_score(y_test, y_pred_test)
                            # Train set
                            precis_train = precision_score(y_train, y_pred_train)
                            rappel_train = recall_score(y_train, y_pred_train)
                            F1_train = f1_score(y_train, y_pred_train)
                            accur_train = accuracy_score(y_train, y_pred_train)
                            session["knn_metrics"] = []
                            session["knn_metrics"].append([round(precis_test, 3), round(precis_test - precis_train, 3)])
                            session["knn_metrics"].append([round(rappel_test, 3), round(rappel_test - rappel_train, 3)])
                            session["knn_metrics"].append([round(F1_test, 3), round(F1_test - F1_train, 3)])
                            session["knn_metrics"].append([round(accur_test, 3), round(accur_test - accur_train, 3)])

                            fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
                            message_ROC = f'Area Under the Curve (AUC) = {round(auc(fpr, tpr), 4)}'
                            fig = px.area(
                                x=fpr, y=tpr,
                                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                                width=500, height=500,
                            )
                            fig.add_shape(
                                type='line', line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1
                            )

                            fig.update_yaxes(scaleanchor="x", scaleratio=1)
                            fig.update_xaxes(constrain='domain')
                            fig.update_layout(
                                font=dict(size=10),
                                autosize=False,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                width=900, height=450,
                                margin=dict(l=40, r=40, b=40, t=40),
                            )
                            figure_roc_knn = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                        # Learning curve
                        N, train_score, val_score = learning_curve(best_model_knn, X_train, y_train,
                                                                   train_sizes=np.linspace(0.2,
                                                                                           1.0,
                                                                                           10),
                                                                   cv=3, random_state=4)
                        fig = go.Figure()
                        fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train',
                                        marker=dict(color='deepskyblue'))
                        fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation',
                                        marker=dict(color='red'))
                        fig.update_layout(
                            showlegend=True,
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        figure_learning_curves_knn = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                        # Faire une prédiction
                        if session['predictions']:
                            try:
                                prediction_knn = best_model_knn.predict(np.array(session['predictions'], dtype=float).reshape(1, -1))
                                message_prediction_knn = f'Prédiction de la target {session["selected_target_knn"][0]} avec les données entrées : {str(df_origine[session["selected_target_knn"][0]].unique()[int(prediction_knn[0])])}'
                            except:
                                message_prediction_knn = "erreur"

                    except:
                        # KNN impossible
                        erreur = True
            else:
                # Dataset vide
                erreur = True

        return render_template("KNN.html", zip=zip, choix_col=choix_col, len=len, erreur=erreur, message_encode=message_encode, meilleur_k_knn=meilleur_k_knn, message_ROC=message_ROC, figure_roc_knn=figure_roc_knn,
                               figure_learning_curves_knn=figure_learning_curves_knn, message_prediction_knn=message_prediction_knn)
    else:
        return render_template("waiting_for_data.html")


@app.route('/KMeans', methods=['GET', 'POST'])
def KMeans_page():
    if '_choix_dataset' in session.keys():
        data = dico_dataset[session['_choix_dataset']]
        figure_kmeans, erreur = None, None
        choix_col = col_numeric(data)
        if 'selected_col_kmeans' not in session:
            session['selected_col_kmeans'] = []
        if 'cluster_kmeans' not in session:
            session['cluster_kmeans'] = None

        if request.method == "POST":
            if request.form.getlist('selected_col_kmeans'):
                session['selected_col_kmeans'] = request.form.getlist('selected_col_kmeans')
            if request.form.getlist('cluster_kmeans'):
                session['cluster_kmeans'] = request.form.getlist('cluster_kmeans')[0]

        if len(session['selected_col_kmeans']) > 1 and session['cluster_kmeans']:
            if int(session['cluster_kmeans']) < 25 and int(session['cluster_kmeans']) > 0:
                df_ml = data[session['selected_col_kmeans']].dropna(axis=0)
                if len(df_ml) > 0:
                    X = df_ml[session['selected_col_kmeans']]  # features
                    #try:
                    # PCA
                    model = PCA(n_components=2)
                    model.fit(X)
                    x_pca = model.transform(X)
                    df = pd.concat([pd.Series(x_pca[:, 0]).reset_index(drop=True),
                                    pd.Series(x_pca[:, 1]).reset_index(drop=True)], axis=1)
                    df.columns = ["x", "y"]

                    # K-Means
                    X_pca_kmeans = df
                    modele = KMeans(n_clusters=int(session['cluster_kmeans']))
                    modele.fit(X_pca_kmeans)
                    y_kmeans = modele.predict(X_pca_kmeans)
                    df["class"] = pd.Series(y_kmeans)

                    fig = px.scatter(df, x=X_pca_kmeans['x'], y=X_pca_kmeans['y'], color="class",
                                     )
                    fig.update_layout(
                        showlegend=True,
                        template='simple_white',
                        font=dict(size=10),
                        autosize=False,
                        width=900, height=450,
                        margin=dict(l=40, r=40, b=40, t=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title="K-Means avec " + str(session['cluster_kmeans']) + " Clusters",
                    )
                    fig.update(layout_coloraxis_showscale=False)
                    centers = modele.cluster_centers_
                    fig.add_scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',
                                    marker=dict(color='black', size=15), opacity=0.5, name='Centroïdes')
                    figure_kmeans = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                    #except:
                        #erreur = True
                else:
                    erreur = True
            else:
                erreur = True

        return render_template("KMeans.html", zip=zip, choix_col=choix_col, erreur=erreur, figure_kmeans=figure_kmeans)
    else:
        return render_template("waiting_for_data.html")


@app.route('/SVM', methods=['GET', 'POST'])
def SVM():
    if '_choix_dataset' in session.keys():
        data = dico_dataset[session['_choix_dataset']]
        figure_svm, erreur = None, None
        choix_col = col_numeric(data)
        all_col = data.columns.values
        if 'selected_features_svm' not in session:
            session['selected_features_svm'] = []
        if 'selected_target_svm' not in session:
            session['selected_target_svm'] = None
        if 'selected_classes_svm' not in session:
            session['selected_classes_svm'] = []

        if request.method == "POST":
            if request.form.getlist('selected_features_svm'):
                session['selected_features_svm'] = request.form.getlist('selected_features_svm')
            if request.form.getlist('selected_target_svm'):
                session['selected_target_svm'] = request.form.getlist('selected_target_svm')[0]
            if request.form.getlist('selected_classes_svm'):
                session['selected_classes_svm'] = request.form.getlist('selected_classes_svm')

        if session['selected_target_svm']:
            choix_classe = data[session['selected_target_svm']].unique().tolist()
        else:
            choix_classe = None

        if len(session['selected_features_svm']) == 2 and session['selected_target_svm'] and session['selected_target_svm'] not in session['selected_features_svm'] and len(session['selected_classes_svm']) == 2:
            target, features = session['selected_target_svm'], session['selected_features_svm']
            df = data[[target] + features]
            df.dropna(axis=0, inplace=True)
            if len(df[target].unique().tolist()) > 1:
                df = df.loc[(df[target] == session['selected_classes_svm'][0]) | (
                            df[target] == session['selected_classes_svm'][1])]
                y = df[target]
                X = df[features]

                fig = px.scatter(df, x=features[0], y=features[1], color=target,
                                 color_continuous_scale=px.colors.diverging.Picnic)
                fig.update(layout_coloraxis_showscale=False)

                model = SVC(kernel='linear', C=1E10)
                model.fit(X, y)

                # Support Vectors
                fig.add_scatter(x=model.support_vectors_[:, 0],
                                y=model.support_vectors_[:, 1],
                                mode='markers',
                                name="Support vectors",
                                marker=dict(size=12,
                                            line=dict(width=1,
                                                      color='DarkSlateGrey'
                                                      ),
                                            color='rgba(0,0,0,0)'),
                                )

                # hyperplan
                w = model.coef_[0]
                a = -w[0] / w[1]
                xx = np.linspace(df[features[0]].min(), df[features[0]].max())
                yy = a * xx - (model.intercept_[0]) / w[1]
                fig.add_scatter(x=xx, y=yy, line=dict(color='black', width=2),
                                name='Hyperplan')

                # Hyperplans up et down
                b = model.support_vectors_[0]
                yy_down = a * xx + (b[1] - a * b[0])
                fig.add_scatter(x=xx, y=yy_down,
                                line=dict(color='black', width=1, dash='dot'),
                                name='Marges')
                b = model.support_vectors_[-1]
                yy_up = a * xx + (b[1] - a * b[0])
                fig.add_scatter(x=xx, y=yy_up,
                                line=dict(color='black', width=1, dash='dot'),
                                showlegend=False)
                fig.update_layout(
                    showlegend=True,
                    template='simple_white',
                    font=dict(size=10),
                    autosize=False,
                    width=900, height=450,
                    margin=dict(l=40, r=40, b=40, t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                figure_svm = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            else:
                erreur = True

        elif session['selected_features_svm'] and session['selected_target_svm'] and session['selected_classes_svm']:
            erreur = True

        return render_template("SVM.html", choix_col=choix_col, all_col=all_col, erreur=erreur, choix_classe=choix_classe,
                               figure_svm=figure_svm,zip=zip)
    else:
        return render_template("waiting_for_data.html")


@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    if '_choix_dataset' in session.keys():
        data = dico_dataset[session['_choix_dataset']]
        figure_tree, erreur, dot_data = None, None, None
        choix_col = col_numeric(data)
        all_col = data.columns.values

        if 'selected_features_tree' not in session:
            session['selected_features_tree'] = []
        if 'selected_target_tree' not in session:
            session['selected_target_tree'] = None

        if request.method == "POST":
            if request.form.getlist('selected_features_tree'):
                session['selected_features_tree'] = request.form.getlist('selected_features_tree')
            if request.form.getlist('selected_target_tree'):
                session['selected_target_tree'] = request.form.getlist('selected_target_tree')[0]

        if len(session['selected_features_tree']) == 2 and session['selected_target_tree'] and session['selected_target_tree'] not in session['selected_features_tree']:
            target = session['selected_target_tree']
            features = session['selected_features_tree']
            df = data[[target] + features].dropna(axis=0)
            if len(df) > 0:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], random_state=0)
                    clf = DecisionTreeClassifier(random_state=0)
                    clf.fit(X_train, y_train)
                    if len(pd.unique(df[target])) > 2:
                        average = 'macro'
                    else:
                        average = 'binary'

                    # metrics on train
                    y_pred_train = clf.predict(X_train)
                    accur_train = accuracy_score(y_train, y_pred_train)
                    precis_train = precision_score(y_train, y_pred_train, average=average)
                    rappel_train = recall_score(y_train, y_pred_train, average=average)
                    F1_train = f1_score(y_train, y_pred_train, average=average)

                    # metrics on test
                    y_pred_test = clf.predict(X_test)
                    accur_test = accuracy_score(y_test, y_pred_test)
                    precis_test = precision_score(y_test, y_pred_test, average=average)
                    rappel_test = recall_score(y_test, y_pred_test, average=average)
                    F1_test = f1_score(y_test, y_pred_test, average=average)

                    session["decision_tree_metrics"] = []
                    session["decision_tree_metrics"].append([round(precis_test, 3), round(precis_test - precis_train, 3)])
                    session["decision_tree_metrics"].append([round(rappel_test, 3), round(rappel_test - rappel_train, 3)])
                    session["decision_tree_metrics"].append([round(F1_test, 3), round(F1_test - F1_train, 3)])
                    session["decision_tree_metrics"].append([round(accur_test, 3), round(accur_test - accur_train, 3)])
                    # DOT data
                    dot_data = export_graphviz(clf, out_file=None,
                                               feature_names=features,
                                               class_names=target,
                                               filled=True,
                                               )
                    # Draw graph
                    # st.graphviz_chart(dot_data, use_container_width=False)

                except:
                    erreur = True

            else:
                erreur = True

        elif session['selected_features_tree'] and session['selected_target_tree']:
            erreur = True

        return render_template("decision_tree.html", choix_col=choix_col, all_col=all_col, figure_tree=figure_tree, erreur=erreur,
                               zip=zip, dot_data=dot_data)
    else:
        return render_template("waiting_for_data.html")


@app.route('/PCA', methods=['GET', 'POST'])
def PCA_page():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        choix_col = df.columns.values
        erreur, message_encode, figure_pca = None, [], None
        if 'selected_col_pca' not in session:
            session['selected_col_pca'] = []
        if 'col_numeriques_pca_avec_encode' not in session:
            session["col_numeriques_pca_avec_encode"] = []
        if 'selected_target_pca' not in session:
            session["selected_target_pca"] = []

        if request.method == "POST":
            if request.form.getlist('selected_col_pca'):
                session['selected_col_pca'] = request.form.getlist('selected_col_pca')
                session['selected_col_encode_pca'], session["col_numeriques_pca_avec_encode"], session[
                    'selected_target_pca'] = [], [], []
                session["col_numeriques_pca_avec_encode"] = col_numeric(df[session['selected_col_pca']])
            if request.form.getlist('selected_col_encode_pca'):
                session['selected_col_encode_pca'] = request.form.getlist('selected_col_encode_pca')[:-1]
                session["col_numeriques_pca_avec_encode"] = list(
                    set(col_numeric(df[session['selected_col_pca']]) + session['selected_col_encode_pca']))
                session['selected_target_pca'] = []
            if request.form.getlist('selected_target_pca'):
                session['selected_target_pca'] = request.form.getlist('selected_target_pca')
                temp = session['selected_col_pca'].copy()
                temp.remove(session['selected_target_pca'][0])

        if len(session['selected_col_pca']) > 1:
            df_ml = df[session['selected_col_pca']]
            df_ml = df_ml.dropna(axis=0)
            df_origine = df_ml.copy()
            if len(df_ml) > 0:
                if session['selected_col_encode_pca']:
                    # Encodage
                    for col in session['selected_col_encode_pca']:
                        message_encode.append(
                            "Colonne " + col + "  :  " + str(df_ml[col].unique().tolist()) + " -> " + str(
                                np.arange(len(df_ml[col].unique()))))
                        df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())), inplace=True)
                if session['selected_target_pca']:
                    try:
                        # PCA
                        model = PCA(n_components=2)
                        y = df_ml[session['selected_target_pca']]  # target
                        X = df_ml.drop(session['selected_target_pca'], axis=1)  # features
                        model.fit(X)
                        x_pca = model.transform(X)
                        # résultats points
                        data = pd.concat([pd.Series(x_pca[:, 0]).reset_index(drop=True),
                                                         pd.Series(x_pca[:, 1]).reset_index(drop=True),
                                                         pd.Series(df_origine[
                                                                       session['selected_target_pca'][0]]).reset_index(
                                                             drop=True)], axis=1)
                        data.columns = ["x", "y", str(session['selected_target_pca'][0])]
                        fig = px.scatter(data, x="x", y="y",
                                         color=str(session['selected_target_pca'][0]),
                                         labels={'color': '{}'.format(str(session['selected_target_pca'][0]))},
                                         color_discrete_sequence=px.colors.qualitative.Plotly)
                        fig.update_layout(
                            showlegend=True,
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        fig.update(layout_coloraxis_showscale=False)
                        figure_pca = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                    except:
                        # PCA impossible
                        erreur = True
            else:
                # Dataset vide
                erreur = True

        return render_template("PCA.html", zip=zip, choix_col=choix_col, len=len, erreur=erreur,
                                   message_encode=message_encode, figure_pca=figure_pca)
    else:
        return render_template("waiting_for_data.html")


@app.route('/UMAP', methods=['GET', 'POST'])
def UMAP_page():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        choix_col = df.columns.values
        erreur, message_encode, figure_umap = None, [], None
        if 'selected_col_umap' not in session:
            session['selected_col_umap'] = []
        if 'col_numeriques_umap_avec_encode' not in session:
            session["col_numeriques_umap_avec_encode"] = []
        if 'selected_target_umap' not in session:
            session["selected_target_umap"] = []

        if request.method == "POST":
            if request.form.getlist('selected_col_umap'):
                session['selected_col_umap'] = request.form.getlist('selected_col_umap')
                session['selected_col_encode_umap'], session["col_numeriques_umap_avec_encode"], session[
                    'selected_target_umap'] = [], [], []
                session["col_numeriques_umap_avec_encode"] = col_numeric(df[session['selected_col_umap']])
            if request.form.getlist('selected_col_encode_umap'):
                session['selected_col_encode_umap'] = request.form.getlist('selected_col_encode_umap')[:-1]
                session["col_numeriques_umap_avec_encode"] = list(
                    set(col_numeric(df[session['selected_col_umap']]) + session['selected_col_encode_umap']))
                session['selected_target_umap'] = []
            if request.form.getlist('selected_target_umap'):
                session['selected_target_umap'] = request.form.getlist('selected_target_umap')
                temp = session['selected_col_umap'].copy()
                temp.remove(session['selected_target_umap'][0])

        if len(session['selected_col_umap']) > 1:
            df_ml = df[session['selected_col_umap']]
            df_ml = df_ml.dropna(axis=0)
            df_origine = df_ml.copy()
            if len(df_ml) > 0:
                if session['selected_col_encode_umap']:
                    # Encodage
                    for col in session['selected_col_encode_umap']:
                        message_encode.append(
                            "Colonne " + col + "  :  " + str(df_ml[col].unique().tolist()) + " -> " + str(
                                np.arange(len(df_ml[col].unique()))))
                        df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())), inplace=True)
                if session['selected_target_umap']:
                    try:
                        # umap
                        model = umap.UMAP()
                        y = df_ml[session['selected_target_umap']]  # target
                        X = df_ml.drop(session['selected_target_umap'], axis=1)  # features
                        model.fit(X)
                        x_umap = model.transform(X)
                        # résultats points
                        data = pd.concat([pd.Series(x_umap[:, 0]).reset_index(drop=True),
                                                         pd.Series(x_umap[:, 1]).reset_index(drop=True),
                                                         pd.Series(df_origine[
                                                                       session['selected_target_umap'][0]]).reset_index(
                                                             drop=True)], axis=1)
                        data.columns = ["x", "y", str(session['selected_target_umap'][0])]
                        fig = px.scatter(data, x="x", y="y",
                                         color=str(session['selected_target_umap'][0]),
                                         labels={'color': '{}'.format(str(session['selected_target_umap'][0]))},
                                         color_discrete_sequence=px.colors.qualitative.Plotly)
                        fig.update_layout(
                            showlegend=True,
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        fig.update(layout_coloraxis_showscale=False)
                        figure_umap = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                    except:
                        # umap impossible
                        erreur = True
            else:
                # Dataset vide
                erreur = True

        return render_template("UMAP.html", zip=zip, choix_col=choix_col, len=len, erreur=erreur,
                                   message_encode=message_encode, figure_umap=figure_umap)
    else:
        return render_template("waiting_for_data.html")


@app.route('/TSNE', methods=['GET', 'POST'])
def TSNE_page():
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        choix_col = df.columns.values
        erreur, message_encode, figure_tsne = None, [], None
        if 'selected_col_tsne' not in session:
            session['selected_col_tsne'] = []
        if 'col_numeriques_tsne_avec_encode' not in session:
            session["col_numeriques_tsne_avec_encode"] = []
        if 'selected_target_tsne' not in session:
            session["selected_target_tsne"] = []

        if request.method == "POST":
            if request.form.getlist('selected_col_tsne'):
                session['selected_col_tsne'] = request.form.getlist('selected_col_tsne')
                session['selected_col_encode_tsne'], session["col_numeriques_tsne_avec_encode"], session[
                    'selected_target_tsne'] = [], [], []
                session["col_numeriques_tsne_avec_encode"] = col_numeric(df[session['selected_col_tsne']])
            if request.form.getlist('selected_col_encode_tsne'):
                session['selected_col_encode_tsne'] = request.form.getlist('selected_col_encode_tsne')[:-1]
                session["col_numeriques_tsne_avec_encode"] = list(
                    set(col_numeric(df[session['selected_col_tsne']]) + session['selected_col_encode_tsne']))
                session['selected_target_tsne'] = []
            if request.form.getlist('selected_target_tsne'):
                session['selected_target_tsne'] = request.form.getlist('selected_target_tsne')
                temp = session['selected_col_tsne'].copy()
                temp.remove(session['selected_target_tsne'][0])

        if len(session['selected_col_tsne']) > 1:
            df_ml = df[session['selected_col_tsne']]
            df_ml = df_ml.dropna(axis=0)
            df_origine = df_ml.copy()
            if len(df_ml) > 0:
                if session['selected_col_encode_tsne']:
                    # Encodage
                    for col in session['selected_col_encode_tsne']:
                        message_encode.append(
                            "Colonne " + col + "  :  " + str(df_ml[col].unique().tolist()) + " -> " + str(
                                np.arange(len(df_ml[col].unique()))))
                        df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())), inplace=True)
                if session['selected_target_tsne']:
                    try:
                        # tsne
                        model = TSNE(n_components=2, random_state=0)
                        y = df_ml[session['selected_target_tsne']]  # target
                        X = df_ml.drop(session['selected_target_tsne'], axis=1)  # features
                        x_tsne = model.fit_transform(X, )
                        # résultats points
                        data = pd.concat([pd.Series(x_tsne[:, 0]).reset_index(drop=True),
                                                         pd.Series(x_tsne[:, 1]).reset_index(drop=True),
                                                         pd.Series(df_origine[
                                                                       session['selected_target_tsne'][0]]).reset_index(
                                                             drop=True)], axis=1)
                        data.columns = ["x", "y", str(session['selected_target_tsne'][0])]
                        fig = px.scatter(data, x="x", y="y",
                                         color=str(session['selected_target_tsne'][0]),
                                         labels={'color': '{}'.format(str(session['selected_target_tsne'][0]))},
                                         color_discrete_sequence=px.colors.qualitative.Plotly)
                        fig.update_layout(
                            showlegend=True,
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=900, height=450,
                            margin=dict(l=40, r=40, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                        )
                        fig.update(layout_coloraxis_showscale=False)
                        figure_tsne = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                    except:
                        # tsne impossible
                        erreur = True
            else:
                # Dataset vide
                erreur = True

        return render_template("TSNE.html", zip=zip, choix_col=choix_col, len=len, erreur=erreur,
                                   message_encode=message_encode, figure_tsne=figure_tsne)
    else:
        return render_template("waiting_for_data.html")


"""
@app.route('/<string:name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

if __name__ == '__main__':
    app.run()
"""
