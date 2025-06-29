import pandas as pd
import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    return X, y

def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    return accuracy

def main():
    X, y = load_data("data/iris.csv")
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    clf = train_model(X_train, y_train)
    accuracy = evaluate_model(clf, X_test, y_test)
    # Save model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(clf, "artifacts/model.joblib")
    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": float(accuracy)}, f)
if __name__ == "__main__":
    main()

