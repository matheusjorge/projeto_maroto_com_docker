import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def data_description(data):
    return data.describe()

def correlation_plot(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.drop(columns=["target"]).corr(), cmap="Blues", annot=True, ax=ax)

    return fig

def eda(data):
    description = data_description(data)
    corr = correlation_plot(data)

    return description, corr