from sklearn.model_selection import train_test_split

def data_split(data):
    X = data.drop(columns=["target"]).copy()
    y = data["target"].copy()

    return train_test_split(X, y, test_size=0.1)

def preprocessing(data):    
    X_train, X_test, y_train, y_test = data_split(data)
    
    return X_train, X_test, y_train, y_test