import pandas as pd
#from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def data():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file('fedesoriano/stroke-prediction-dataset', file_name = 'healthcare-dataset-stroke-data.csv')
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return data


def drop_na(df) -> pd.DataFrame:
    newData = df.dropna()
    return newData


def make_binary(df):
    data_cols = ['gender', 'ever_married']
    for data_col in data_cols:
        unique = df[data_col].unique().tolist()
        for index, value in df[data_col].items():
            if value in unique:
                df.at[index, data_col] = int(unique.index(value))
        df[data_col] = df[data_col].astype(int)
    return df


def remove_cols(df: pd.DataFrame) -> pd.DataFrame:
    col_list = ["id", "smoking_status", "Residence_type", "work_type"]
    data = df.drop(col_list, axis = 1)
    return data


def filter_adults(df) -> pd.DataFrame:
    filter = df['age'] >= 18
    return df[filter]

def logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state= 4)
    model = LogisticRegression(solver='liblinear', random_state=0, class_weight = 'balanced')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    return model, y_test, y_pred, acc

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
print(data)

data = drop_na(data)
data = make_binary(data)
data = remove_cols(data)
data = filter_adults(data)
data = data[data.gender != 2]
X = data.drop(columns = ["stroke"])
y = data["stroke"]
model, y_test, y_pred, acc = logistic_regression(X, data["stroke"])
print(acc)
print(roc_auc_score(y, model.predict_proba(X)[:, 1]))

#plot_cm(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
disp.plot()
plt.savefig('confusion_matrix.png')