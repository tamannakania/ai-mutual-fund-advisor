# backend/ml_models/train_models.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data(path="backend/datasets/cleaned_mutual_funds.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, target="Return"):
    # Encode categorical features
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, "backend/ml_models/label_encoders.pkl")
    return df

def train_return_model(df):
    X = df.drop(columns=["Return"])
    y = df["Return"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, "backend/ml_models/best_return_model.pkl")

    preds = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"RMSE": rmse, "Best Params": grid.best_params_}

def train_risk_classifier(df):
    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "backend/ml_models/risk_classifier.pkl")

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {"Accuracy": acc}
