from data_collect import data_collect
from utils import load_model, save_data
import os
import pandas as pd

if __name__ == "__main__":
    # Ir pro diretório pai
    os.chdir("..")

    # Coleta de dados
    data = data_collect()
    X_pred = data.drop(columns=["target"]).to_numpy()

    # Carregar modelo
    model = load_model("models/model.pkl")

    preds = model.predict(X_pred)
    save_data(pd.Series(preds), "data/prediction.csv")