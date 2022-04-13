from sklearn.metrics import accuracy_score

def evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)