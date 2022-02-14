import numpy as np
import pandas as pd
from scipy.spatial import distance

def max_std(dataset):  # colonne de maximum de variance
    l = []
    for nom in dataset.columns:
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str:
            l.append([dataset[nom].std(), nom])
    return max(l)


def col_numeric(df):  # retourne les colonnes numériques d'un dataframe
    return df.select_dtypes(include=np.number).columns.tolist()


def col_temporal(df):  # retourne les colonnes temporelles d'un dataframe
    return df.select_dtypes(include=np.datetime64).columns.tolist()


def clean_data(x):  # enlever les symboles d'une colonne
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '').replace('€', '').replace('£', '')
    return x


def distance_e(x, y):  # distance entre 2 points du plan cartésien
    return distance.euclidean([x[0], x[1]], [y[0], y[1]])


def max_dist(donnee_apres_pca, df, voisins):  # pour knn, retourne la distance du voisin le plus loin
    distances = []
    for i in range(len(df)):
        distances.append(distance_e(donnee_apres_pca, [df['x'].iloc[i], df['y'].iloc[i]]))
    distances.sort()
    return distances[voisins-1]

def type_col_dataset(df):
    types_col = [str(df.dtypes.value_counts().index[i]) for i in range(len(df.dtypes.value_counts()))]
    num_col_types = [str(df.dtypes.value_counts().values[i]) for i in range(len(df.dtypes.value_counts()))]
    res = ""
    for i, j in zip(types_col, num_col_types):
        res += f"{i} -> {j}  \n"
    return res

def all_caract(df):
    return {
                'taille': df.shape,
                'nombre_de_val': str(df.shape[0] * df.shape[1]),
                'type_col': type_col_dataset(df),
                'pourcentage_missing_val': [
                    str(round(sum(df.isnull().sum(axis=1).tolist()) * 100 / (df.shape[0] * df.shape[1]), 2)),
                    str(sum(df.isnull().sum(axis=1).tolist()))]
            }