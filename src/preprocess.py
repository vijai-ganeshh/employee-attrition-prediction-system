import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()

    # Target variable
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Selected features from IBM dataset
    features = [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "YearsAtCompany",
        "WorkLifeBalance",
        "OverTime",
        "Gender"
    ]

    X = df[features]
    y = df["Attrition"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, features
