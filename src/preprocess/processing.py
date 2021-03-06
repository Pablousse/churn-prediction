import pandas as pd
from sklearn.model_selection import train_test_split
import types
from imblearn.over_sampling import ADASYN, SMOTE


def initialize_dataframe() -> object:
    """Initialize the dataframe to train and test the model

    Returns:
        a data object composed of X,y train and test data
    """

    df = pd.read_csv("../assets/BankChurners.csv")

    data = split_data(df, "Attrition_Flag")

    data.X_train, data.y_train = train_pipeline(data.X_train, data.y_train)
    data.X_test, data.y_test = train_pipeline(data.X_test, data.y_test)

    return data


def train_pipeline(X, y):
    """Pipeline function to format the dataframe for the model

    Args:
        X : the features
        y : the target
    """
    drop_columns(X)
    one_hot_encode_feature(X)
    X = label_encode(X)
    one_hot_encode_target(y)

    return X, y


def split_data(df: pd.DataFrame, feature_name: str) -> object:
    """Split the data into train and test set for the model to train

    Args:
        df :(pd.DataFrame) dataframe
        feature_name (str): the name of the feature column in the dataframe

    Returns:
        a data object composed of X,y train and test data
    """
    X = df.drop(feature_name, axis=1)
    y = df[feature_name]

    data = types.SimpleNamespace()

    data.X_train, data.X_test, data.y_train, data.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return data


def drop_columns(df: pd.DataFrame) -> None:
    """Drop the column that are useless for the model

    Args:
        df (pd.DataFrame): The dataframe
    """

    df.drop(
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category"
        "_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        axis=1,
        inplace=True,
    )
    df.drop(
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_"
        "Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        axis=1,
        inplace=True,
    )
    df.drop(
        "Avg_Open_To_Buy",
        axis=1,
        inplace=True,
    )
    df.drop("CLIENTNUM", axis=1, inplace=True)


def one_hot_encode_feature(df: pd.DataFrame) -> None:
    """Custom one hot encoder

    Args:
        df (pd.DataFrame): dataframe
    """
    df["Gender"].replace("F", "0", inplace=True)
    df["Gender"].replace("M", "1", inplace=True)

    df.rename(columns={"Gender": "is_male"}, inplace=True)

    df["is_male"] = df["is_male"].astype(int)


def one_hot_encode_target(y: pd.Series) -> None:
    """One hot encoder for the feature

    Args:
        y (pd.Series): the feature
    """
    y.replace("Existing Customer", "0", inplace=True)
    y.replace("Attrited Customer", "1", inplace=True)

    y.rename("has_left", inplace=True)

    y = y.astype(int)


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Custom label encoder

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        df (pd.DataFrame): dataframe
    """
    cleanup_nums = {
        "Education_Level": {
            "Unknown": 0,
            "Uneducated": 1,
            "High School": 2,
            "College": 3,
            "Graduate": 4,
            "Post-Graduate": 5,
            "Doctorate": 6,
        },
        "Income_Category": {
            "Unknown": 0,
            "Less than $40K": 1,
            "$40K - $60K": 2,
            "$60K - $80K": 3,
            "$80K - $120K": 4,
            "$120K +": 5,
        },
        "Card_Category": {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3},
    }

    df.replace(cleanup_nums, inplace=True)

    df = pd.get_dummies(df, columns=["Marital_Status"])

    return df


def apply_over_sampling(data: object, over_sampling_type: str) -> object:
    """A function to apply over sampling in case of an imbalanced dataset

    Args:
        data (object): custom data object composed of X,y train and test data
        over_sampling_type (str): can be SMOTE or ADASYN, define the type of over sampling

    Returns:
        object: custom data object
    """
    if over_sampling_type == "SMOTE":
        over_sampling = SMOTE(random_state=42)
    elif over_sampling_type == "ADASYN":
        over_sampling = ADASYN(random_state=42)

    new_data = types.SimpleNamespace()

    new_data.X_train, new_data.y_train = over_sampling.fit_resample(data.X_train, data.y_train)
    new_data.X_test = data.X_test
    new_data.y_test = data.y_test

    return new_data
