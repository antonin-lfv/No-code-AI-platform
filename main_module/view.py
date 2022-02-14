# Importations
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
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
    'Iris (Classification)': pd.read_csv("main_module/static/datasets/iris.csv"),
    'Penguins (Classification)': pd.read_csv("main_module/static/datasets/penguins.csv"),
    'Prix des voitures (Régression)': pd.read_csv("main_module/static/datasets/CarPrice.csv"),
}

# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
def home():
    # pour clear les objets enregistrés sur la session dès qu'on revient au menu
    # session.clear()
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


@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    select_dataset = SelectDataset()
    if select_dataset.validate_on_submit():
        _choix_dataset = select_dataset.choix.data
        session['_choix_dataset'] = _choix_dataset
        if session['_choix_dataset'] in dico_dataset.keys():
            df = dico_dataset[session['_choix_dataset']]
            caract_dataset = all_caract(df)
            return render_template('dataset.html', select_dataset=select_dataset, column_names=df.columns.values,
                                   row_data=list(df.values.tolist()), zip=zip, caract_dataset=caract_dataset)
        else:
            return render_template('dataset.html', select_dataset=select_dataset)
    else:
        if session['_choix_dataset'] in dico_dataset.keys():
            df = dico_dataset[session['_choix_dataset']]
            caract_dataset = all_caract(df)
            return render_template('dataset.html', select_dataset=select_dataset, column_names=df.columns.values,
                                   row_data=list(df.values.tolist()), zip=zip, caract_dataset=caract_dataset)
        else:
            return render_template('dataset.html', select_dataset=select_dataset)


@app.route('/analyse_colonnes', methods=['GET', 'POST'])
def analyse_colonnes():
    # return render_template('analyse_colonnes.html')
    if '_choix_dataset' in session.keys():
        df = dico_dataset[session['_choix_dataset']]
        return render_template("analyse_colonnes.html", column_names=df.columns.values,
                               row_data=list(df.values.tolist()),
                               zip=zip)
    else:
        return render_template("analyse_colonnes.html")


@app.route('/matrice_correlation', methods=['GET', 'POST'])
def matrice_correlation():
    return render_template('matrice_correlation.html')


@app.route('/section_graphique', methods=['GET', 'POST'])
def section_graphiques():
    return render_template('section_graphiques.html')


@app.route('/regressions', methods=['GET', 'POST'])
def regressions():
    return render_template('regressions.html')


@app.route('/KNN', methods=['GET', 'POST'])
def KNN():
    return render_template('KNN.html')


@app.route('/KMeans', methods=['GET', 'POST'])
def KMeans():
    return render_template('KMeans.html')


@app.route('/SVM', methods=['GET', 'POST'])
def SVM():
    return render_template('SVM.html')


@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    return render_template('decision_tree.html')


@app.route('/PCA', methods=['GET', 'POST'])
def PCA():
    return render_template('PCA.html')


@app.route('/UMAP', methods=['GET', 'POST'])
def UMAP():
    return render_template('UMAP.html')


@app.route('/TSNE', methods=['GET', 'POST'])
def TSNE():
    return render_template('TSNE.html')


"""
@app.route('/<string:name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

if __name__ == '__main__':
    app.run()
"""
