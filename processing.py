import pandas as pd


def initialize_dataframe():

    df = pd.read_csv("assets/BankChurners.csv")

    df = df.drop(
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category"
        "_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        axis=1,
    )
    df = df.drop(
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_"
        "Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        axis=1,
    )
    df = df.drop("CLIENTNUM", axis=1)

    df["Attrition_Flag"].replace("Existing Customer", "0", inplace=True)
    df["Attrition_Flag"].replace("Attrited Customer", "1", inplace=True)

    df.rename(columns={"Attrition_Flag": "has_left"}, inplace=True)

    df["Gender"].replace("F", "0", inplace=True)
    df["Gender"].replace("M", "1", inplace=True)

    df.rename(columns={"Gender": "is_male"}, inplace=True)

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

    df = df.replace(cleanup_nums)

    df = pd.get_dummies(df, columns=["Marital_Status"])

    df["has_left"] = df["has_left"].astype(int)
    df["is_male"] = df["is_male"].astype(int)

    return df
