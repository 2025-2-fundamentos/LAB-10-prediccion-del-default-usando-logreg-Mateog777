import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
import gzip
import os
import json
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


def load_data(csv_file):
    df = pd.read_csv(csv_file, compression="zip")
    return df

#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".

def data_clean(data):
    df = data.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns="ID", inplace=True)
    df = df[(df["EDUCATION"]!=0) & (df["MARRIAGE"]!=0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x>4 else x)
    return df

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
def split_data(data_train, data_test):
    x_train = data_train.drop(columns="default")
    y_train = data_train["default"]
    x_test = data_test.drop(columns="default")
    y_test = data_test["default"]
    return x_train, y_train, x_test, y_test

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.

def create_pipeline():
    categorical_feature=['EDUCATION','SEX','MARRIAGE']


    # Crear el transformador para las columnas categóricas
    preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_feature)],remainder=MinMaxScaler())

    # Crear el pipeline con preprocesamiento y el modelo
    pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_regression, k=10)), # Seleccionar las 10 mejores características/vairables
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])


    return pipeline

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
def make_grid_search(pipeline):

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid = {
            'feature_selection__k':range(1,11),
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty':['l1','l2'],
            'classifier__solver':['liblinear'],
            "classifier__max_iter": [100,200],
        },
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )

    return grid_search

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
def save_model(estimator, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

def check_estimator(estimator, x, y, dataset):
    y_pred = estimator.predict(x)

    precision = round(precision_score(y, y_pred), 4)
    balanced_accuracy = round(balanced_accuracy_score(y, y_pred), 4)
    f1 = round(f1_score(y, y_pred), 4)
    recall = round(recall_score(y, y_pred), 4)

    metrics = {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "f1_score": f1
    }
    
    return metrics, y_pred, y

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
def c_matrix(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix", "dataset": dataset,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }
    
def main():
    os.makedirs("files/output", exist_ok=True)

    # Cargar los datos
    df_train = data_clean(load_data('files/input/train_data.csv.zip'))
    df_test = data_clean(load_data('files/input/test_data.csv.zip'))

    # Dividir los datos
    x_train, y_train, x_test, y_test = split_data(df_train, df_test)

    # Crear el pipeline
    pipeline = create_pipeline()

    # Realizar la búsqueda de hiperparámetros al pipeline
    grid_search = make_grid_search(pipeline)

    # Entrenar el modelo con los hipérparámetros optimizados
    estimador = grid_search.fit(x_train, y_train)

    # Metricas de entrenamiento y prueba
    metrics_train, y_pred_train, y_train = check_estimator(estimador, x_train, y_train, "train")
    metrics_test, y_pred_test, y_test = check_estimator(estimador, x_test, y_test, "test")

    # Matrices de confusión de entrenamiento y prueba
    c_train = c_matrix(y_train, y_pred_train, "train")
    c_test = c_matrix(y_test, y_pred_test, "test")

    with open("files/output/metrics.json", "w") as file:
            file.write(json.dumps(metrics_train) + "\n")
            file.write(json.dumps(metrics_test) + "\n")
            file.write(json.dumps(c_train) + "\n")
            file.write(json.dumps(c_test) + "\n")

    save_model(estimador, "files/models/model.pkl.gz")

main()