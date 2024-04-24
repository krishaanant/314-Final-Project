import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, y_test, y_pred, acc, score

# def plot_cm(y_test, y_pred):
#     cm = confusion_matrix(y_test, y_pred)
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(cm)
#     ax.grid(False)
#     ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
#     ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
#     ax.set_ylim(1.5, -0.5)
#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
#     plt.show()

def plot_cm(y_test, y_pred, classes):
    plt.clf()
    cm = confusion_matrix(y_test, y_pred, labels = classes)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
    disp.plot()
    plt.savefig('confusion_matrix.png')

data = data()
# print(data)

# preprocess data 
data = drop_na(data)
data = make_binary(data)
data = remove_cols(data)
data = filter_adults(data)
data = data[data.gender != 2]

# train a logistic regression model 
X = data.drop(columns = ["stroke"])
y = data["stroke"]
model, y_test, y_pred, acc, roc_auc_score = logistic_regression(X, data["stroke"])
print("Accuracy: ", acc)
print("ROC/AUC score: ", roc_auc_score)

# plot confusion matrix 
#plot_cm(y_test, y_pred)
plot_cm(y_test, y_pred, model.classes_)
plt.savefig('confusion_matrix.png')

# train PCA (2 principal components)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components = 2, random_state = 4)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

# graph PCA result using top two principal components
plt.clf()
stroke = X_pca[y==1]
non_stroke = X_pca[y==0]
plt.scatter(non_stroke[:, 0], non_stroke[:,1], c = 'grey', alpha = 0.2, label = 'non-stroke')
plt.scatter(stroke[:, 0], stroke[:, 1], c = 'b', alpha = 1, label = 'stroke')
plt.legend()
plt.title("PCA on Stroke Prediction Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("pca.png")

# grab the most important feature from the first principal component
features = model.feature_names_in_
important_features = [features[i] for i in np.argsort(pca.components_[0])][::-1]
print("Two Most Important Features from the First Principal Component: ", important_features[:2])

# graph with the two most important features from the first principal component (which explains the most variance)
plt.clf()
feature1 = important_features[0]
feature2 = important_features[1]
plt.scatter(X[feature1][y==0], X[feature2][y==0], c = 'grey', alpha = 0.2, label = 'non-stroke')
plt.scatter(X[feature1][y==1], X[feature2][y==1], c = 'b', alpha = 1, label = 'stroke')
plt.legend()
plt.title("{a} vs {b}".format(a = feature1, b = feature2))
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.savefig("two_features.png")