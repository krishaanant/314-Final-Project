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

def logistic_regression(x, y):
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x,y)
    model.predict(x)





