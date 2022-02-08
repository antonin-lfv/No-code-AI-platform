# Importations
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, abort, render_template, session, flash
from flask_bootstrap import Bootstrap

from main_module.static.utils import *
from main_module.static.classes import *

app = Flask(__name__)
app.config.from_object('config')
bootstrap = Bootstrap(app)

# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
def home():
    # pour clear les objets enregistrés sur la session dès qu'on revient au menu
    session.clear()
    return render_template('home.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    # form #
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        session['name'] = name
    # dropdown select #
    select = SelectBox()
    if select.validate_on_submit():
        choix = select.choix.data
        session['choix'] = choix
    return render_template('test.html', form=form, select=select)


@app.route('/Dataset', methods=['GET', 'POST'])
def Dataset():
    select_dataset = SelectDataset()
    if select_dataset.validate_on_submit():
        _choix_dataset = select_dataset.choix.data
        session['_choix_dataset'] = _choix_dataset
        dico_dataset = {
            'Iris (Classification)': pd.read_csv("main_module/static/datasets/iris.csv"),
            'Penguins (Classification)': pd.read_csv("main_module/static/datasets/penguins.csv"),
            'Prix des voitures (Régression)': pd.read_csv("main_module/static/datasets/CarPrice.csv"),
        }
        if session['_choix_dataset'] in dico_dataset.keys():
            df = dico_dataset[session['_choix_dataset']]
            caract_dataset = {
                'taille': df.shape,
                'nombre_de_val': str(df.shape[0] * df.shape[1]),
                'type_col': type_col_dataset(df),
                'pourcentage_missing_val': [str(round(sum(df.isnull().sum(axis=1).tolist()) * 100 / (df.shape[0] * df.shape[1]), 2)), str(sum(df.isnull().sum(axis=1).tolist()))]
            }
            return render_template('Dataset.html', select_dataset=select_dataset, column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip, caract_dataset=caract_dataset)
        else:
            return render_template('Dataset.html', select_dataset=select_dataset)
    else:
        return render_template('Dataset.html', select_dataset=select_dataset)


@app.route('/Analyse des colonnes', methods=['GET', 'POST'])
def Analyse_colonnes():
    # return render_template('analyse_colonnes.html')
    return render_template("analyse_colonnes.html", column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip)


@app.route('/Matrice de corrélations', methods=['GET', 'POST'])
def Matrice_correlation():
    return render_template('matrice_correlation.html')


@app.route('/Section graphiques', methods=['GET', 'POST'])
def Section_graphiques():
    return render_template('section_graphiques.html')


@app.route('/Régressions', methods=['GET', 'POST'])
def Regressions():
    return render_template('regressions.html')


"""
@app.route('/<string:name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

if __name__ == '__main__':
    app.run()
"""
