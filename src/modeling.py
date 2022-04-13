import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def generate_models():
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]
    )

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]
    )

    decision_tree = Pipeline(
        [
            ("model", DecisionTreeClassifier())
        ]
    )

    random_forest = Pipeline(
        [
            ("model", RandomForestClassifier())
        ]
    )

    return [logistic_regression, svm, decision_tree, random_forest]

def model_search(models, X_train, y_train):
    scores = []
    for model in models:
        scores.append(cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean())
        
    return models[np.array(scores).argmax()]

def modeling(X_train, y_train):
    models = generate_models()
    model = model_search(models, X_train, y_train)
    
    return model.fit(X_train, y_train)