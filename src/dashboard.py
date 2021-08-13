from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_pie(column_name: str, df: pd.DataFrame) -> Figure:
    """Create a pie plot of a specific feature for the dashboard

    Args:
        column_name (str): the feature to plot
        df (pd.DataFrame): The dataset

    Returns:
        Figure: the plot
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(column_name)

    data = df.groupby(column_name).count().iloc[:, 0]
    data_Attrited = (
        df[df["Attrition_Flag"] == "1"]
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

    return fig


def plot_hist(column_name: str, df: pd.DataFrame) -> Figure:
    """Create a histogram plot of a specific feature for the dashboard

    Args:
        column_name (str): the feature to plot
        df (pd.DataFrame): The dataset

    Returns:
        Figure: the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.title.set_text("All")
    ax2.title.set_text("Attrited Customer")

    bins = np.histogram_bin_edges(df[column_name], bins=20)

    sns.histplot(data=df, x=df[column_name], ax=ax1, bins=bins)
    df_Attrited = df[df["Attrition_Flag"] == "1"]
    sns.histplot(data=df_Attrited , x=df_Attrited[column_name], ax=ax2, bins=bins)

    return fig


bank_churner = Path(__file__).parents[1] / 'assets/BankChurners.csv'

# df = df = pd.read_csv("../assets/BankChurners.csv")
df = df = pd.read_csv(bank_churner)

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

df.rename(columns={"Total_Trans_Ct" : "Total transaction count",
                   "Total_Trans_Amt" : "Total transaction amount",
                   "Total_Amt_Chng_Q4_Q1" : "Change in Transaction Count",
                   "Total_Revolving_Bal" : "Total Revolving Balance",
                   "Avg_Utilization_Ratio" : "Average Utilization Ratio",
                   "Total_Relationship_Count" : "Total Relationship Count",
                   }, inplace=True)

st.title("Churn prediction dashboard")
# st.pyplot(plot_pie("Gender", df))
# st.pyplot(plot_pie("Card_Category", df))

# fig, ax = plt.subplots(figsize=(10, 5))
# matrix = np.triu(df.corr(method="spearman"))
# sns.diverging_palette(220, 20, as_cmap=True)
# sns.heatmap(df.corr(), ax=ax, annot=True, mask=matrix, cmap='BrBG')
# st.write(fig)

fig = plot_hist("Total transaction amount", df)
st.write(fig)

fig = plot_hist("Total transaction count", df)
st.write(fig)

fig = plot_hist("Change in Transaction Count", df)
st.write(fig)

fig = plot_hist("Total Revolving Balance", df)
st.write(fig)

fig = plot_hist("Average Utilization Ratio", df)
st.write(fig)

fig = plot_hist("Total Relationship Count", df)
st.write(fig)
