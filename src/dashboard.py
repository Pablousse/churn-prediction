import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from processing import initialize_dataframe


@st.cache
def load_data():
    return initialize_dataframe()


def plot_pie(column_name, dataframe):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(column_name)

    data = dataframe.groupby(column_name).count().iloc[:, 0]
    data_Attrited = (
        dataframe[dataframe["Attrition_Flag"] == "Attrited Customer"]
        .groupby(column_name)
        .count()
        .iloc[:, 0]
    )
    data.name = column_name
    labels = data.index
    labels_Attrited = data_Attrited.index
    ax1.title.set_text("All")
    ax2.title.set_text("Attrited Customer")
    ax1.pie(data.T, autopct="%.1f%%", startangle=90, labels=labels)
    ax2.pie(data_Attrited.T, autopct="%.1f%%", startangle=90, labels=labels_Attrited)

    # plt.show()
    return fig


df = df = pd.read_csv("assets/BankChurners.csv")

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

st.title("Food Demand Forecasting — Analytics Vidhya")
st.pyplot(plot_pie("Gender", df))
st.pyplot(plot_pie("Card_Category", df))
