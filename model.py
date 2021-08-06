from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from processing import initialize_dataframe
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = initialize_dataframe()


models = [RandomForestClassifier(n_estimators=100), KNeighborsClassifier(n_neighbors=3)]

# model = RandomForestClassifier(n_estimators=100)

X = df.drop("has_left", axis=1)
y = df["has_left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def run_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report: ")
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))
    print(confusion_matrix(y_train, model.predict(X_train)))
    print(confusion_matrix(y, model.predict(X)))


for model in models:
    print("------------------------------------")
    print("model = " + str(model))
    run_model(model)
