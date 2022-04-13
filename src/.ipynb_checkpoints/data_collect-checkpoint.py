from sklearn.datasets import load_wine
import pandas as pd

def data_collect():
    wine_dataset = load_wine()
    data = pd.DataFrame(wine_dataset["data"], columns=wine_dataset["feature_names"])
    data["target"] =  wine_dataset["target"]
    
    return data