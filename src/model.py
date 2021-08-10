import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from processing import initialize_dataframe

df = initialize_dataframe()


models = [
    RandomForestClassifier(n_estimators=100),
    KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(max_depth=14),
    GradientBoostingClassifier(
        n_estimators=1500, learning_rate=1, max_features=10, max_depth=2, random_state=0
    ),
    XGBClassifier(
        colsample_bytree=0.9, learning_rate=0.2, max_depth=7, label_encoder=False
    ),
    lgb.LGBMClassifier(
        max_depth=-1,
        random_state=42,
        silent=True,
        metric="None",
        n_jobs=5,
        n_estimators=1000,
    ),
    ExtraTreesClassifier(),
]

X = df.drop("has_left", axis=1)
y = df["has_left"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class Data:
    pass


data = Data()
data.X_train = X_train
data.X_test = X_test
data.y_train = y_train
data.y_test = y_test

sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X, y)
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
    X_smote, y_smote, test_size=0.2, random_state=42
)

data_smote = Data()
data_smote.X_train = X_train_smote
data_smote.X_test = X_test_smote
data_smote.y_train = y_train_smote
data_smote.y_test = y_test_smote

ada = ADASYN(random_state=42)
X_adasyn, y_adasyn = ada.fit_resample(X, y)
X_train_adasyn, X_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(
    X_adasyn, y_adasyn, test_size=0.2, random_state=42
)

data_adasyn = Data()
data_adasyn.X_train = X_train_adasyn
data_adasyn.X_test = X_test_adasyn
data_adasyn.y_train = y_train_adasyn
data_adasyn.y_test = y_test_adasyn


def plot_feature_importance():

    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    feature_names = [f"feature {column}" for column in X.columns]

    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances.nlargest(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Important Features")
    plt.show()


def run_model(model, data):

    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)

    print("Classification Report: ")
    print(classification_report(data.y_test, y_pred))
    print("confusion matrix test")
    print(confusion_matrix(data.y_test, y_pred))
    print("confusion matrix train")
    print(confusion_matrix(data.y_train, model.predict(data.X_train)))
    print("Cohen Kappa score")
    print(cohen_kappa_score(data.y_test, y_pred))


for model in models:
    print("------------------------------------")
    print("model = " + str(model))
    run_model(model, data)

print("------------------------------------")
print("SMOTE")

for model in models:
    print("------------------------------------")
    print("model = " + str(model))
    run_model(model, data_smote)

print("------------------------------------")
print("ADASYN")

for model in models:
    print("------------------------------------")
    print("model = " + str(model))
    run_model(model, data_adasyn)
