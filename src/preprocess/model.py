from preprocess.processing import apply_over_sampling, initialize_dataframe
import lightgbm as lgb
import joblib


def create_model():
    """Create the churn model
    """
    data = initialize_dataframe()

    data_smote = apply_over_sampling(data, "SMOTE")

    model = lgb.LGBMClassifier(
        max_depth=-1,
        random_state=42,
        silent=True,
        metric="None",
        n_jobs=5,
        n_estimators=1000,
    )

    model.fit(data_smote.X_train, data_smote.y_train)

    return model


def serialize_model(model):
    """Serialize the model into a .pkl file to be able to re-use
    it later

    Args:
        model : the model to serialize
    """
    joblib.dump(model, "model.pkl")
