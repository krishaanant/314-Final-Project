import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def data():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('fedesoriano/stroke-prediction-dataset', file_name = 'healthcare-dataset-stroke-data.csv')
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return data


def drop_na(data) -> pd.DataFrame:
    newData = data.dropna()
    return newData


def make_binary(drop_na):
    data_cols = ['gender', 'ever_married']
    for data_col in data_cols:
        unique = drop_na[data_col].unique().tolist()
        for index, value in drop_na[data_col].items():
            if value in unique:
                drop_na.at[index, data_col] = int(unique.index(value))
        drop_na[data_col] = drop_na[data_col].astype(int)
    return drop_na


def remove_cols(make_binary: pd.DataFrame) -> pd.DataFrame:
    col_list = ["id", "smoking_status"]
    data = make_binary.drop(col_list, axis = 1)
    return data


def filter_adults(remove_cols : pd.DataFrame) -> pd.DataFrame:
    filter = remove_cols['age'] >= 18
    return remove_cols[filter]

def logistic_regression(X, y):
    X_train, X_test, y_train, y_test = test_train_split(X, y, test_size = 0.2, stratify = y, random_state= 4)
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    return y_test, y_pred, acc

def plot_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


data = data()
