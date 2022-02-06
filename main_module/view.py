# Importations
import itertools
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import graphviz
from flask import Flask, request, redirect, abort, render_template
from collections import Counter
from main_module.static.utils import *
import more_itertools

app = Flask(__name__)
app.config.from_object('config')


# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Dataset')
def Dataset():
    return render_template('dataset.html')

df = pd.read_csv("main_module/static/datasets/iris.csv")

@app.route('/Analyse des colonnes')
def Analyse_colonnes():
    # return render_template('analyse_colonnes.html')
    return render_template("analyse_colonnes.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                           link_column="variety", zip=zip)


@app.route('/Matrice de corrélations')
def Matrice_correlation():
    return render_template('matrice_correlation.html')


@app.route('/Section graphiques')
def Section_graphiques():
    return render_template('section_graphiques.html')


@app.route('/Régressions')
def Regressions():
    return render_template('regressions.html')


@app.route('/Classifications')
def Classification():
    return render_template('classification.html')


@app.route('/Ensemble learning')
def Ensemble_learning():
    return render_template('ensemble_learning.html')


@app.route('/Réduction de dimension')
def Reduction_dimension():
    return render_template('reduction_dimension.html')


"""
@app.route('/<string:name>')
def user(name):
    return '<h1>Hello, {}!</h1>'.format(name)

if __name__ == '__main__':
    app.run()
"""
