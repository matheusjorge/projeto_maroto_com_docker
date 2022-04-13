from data_collect import data_collect
from eda import eda
from preprocessing import preprocessing
from modeling import modeling
from evaluation import evaluation
from utils import *

import numpy as np
import os
import pandas as pd

if __name__ == "__main__":
    # Ir para o diretório pai
    os.chdir("..")
    
    # Data collect
    data = data_collect()
    save_data(data, "data/raw_data.csv")
    print(data)
    
    # EDA
    description, corr = eda(data)
    save_data(description, "data/description.csv")
    save_plot(corr, "plots/corr.png")
    print(description)
    
    # Pré-processmento
    X_train, X_test, y_train, y_test = preprocessing(data)
    save_data(pd.DataFrame(X_train), "data/X_train.csv")
    save_data(pd.DataFrame(X_test), "data/X_test.csv")
    save_data(pd.Series(y_train), "data/y_train.csv")
    save_data(pd.Series(y_test), "data/y_test.csv")
    print(X_train.shape, X_test.shape)
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))
    
    # Modelagem
    model = modeling(X_train, y_train)
    save_model(model, "models/model.pkl")
    
    # Avaliação
    acc = evaluation(model, X_test, y_test)
    print(acc)