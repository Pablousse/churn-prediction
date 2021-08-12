import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from rfpimp import permutation_importances


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


class Data:
    pass


def get_models_metrics(data, over_sapmling_type=""):

    if over_sapmling_type != "":
        print(over_sapmling_type)

    for model in models:
        print("------------------------------------")
        print("model = " + str(model))
        run_model(model, data)

        print("------------------------------------")


def plot_feature_importance_random_forest(data):

    forest = RandomForestClassifier(random_state=0)
    forest.fit(data.X_train, data.y_train)

    importances = forest.feature_importances_
    feature_names = [f"feature {column}" for column in data.X_train.columns]
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances.sort_values().plot(kind="barh")
    plt.title("Top 10 Important Features")
    plt.show()


def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))


def plot_permutation_importances(data, model):

    importances = permutation_importances(model, data.X_train, data.y_train, r2)
    importances.sort_values(by="Importance").plot(kind="barh")
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
