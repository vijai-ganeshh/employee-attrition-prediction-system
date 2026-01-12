import pandas as pd
from src.preprocess import preprocess_data
from src.model import logistic_model, random_forest_model

def train_models():
    df = pd.read_csv("data/employee_attrition.csv")

    X, y, scaler, features = preprocess_data(df)

    log_model = logistic_model()
    rf_model = random_forest_model()

    log_model.fit(X, y)
    rf_model.fit(X, y)

    return log_model, rf_model, scaler, X, y, features
