from utils import load_model, save_data
from monitoring import ks_test
import os
import pandas as pd

if __name__ == "__main__":
    # Ir pro diretório pai
    os.chdir("..")

    print(os.getcwd())

    # Coleta de dados
    X_test = pd.read_csv("data/X_test.csv")
    X_train = pd.read_csv("data/X_train.csv")

    for i in range(X_test.shape[1]):
        ks_stat, p_value = ks_test(X_train.iloc[:,i], X_test.iloc[:, i], alpha=0.01)
        print(f"Coluna: {X_test.columns[i]}")
        print(f"\t KS: {ks_stat:.3f} | P Valor: {p_value:.3f}")

    # Carregar modelo
    model = load_model("models/model.pkl")

    preds = model.predict(X_test)
    save_data(pd.Series(preds), "data/prediction.csv")