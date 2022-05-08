import matplotlib.pyplot as plt
import pandas as pd
import pickle

def save_data(data, path):
    data.to_csv(path, index=False)

def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)

def load_model(path):
    with open(path, "rb") as file:
        model = pickle.load(file)

    return model

def save_plot(fig, path):
    fig.savefig(path)