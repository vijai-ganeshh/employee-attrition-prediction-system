from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def logistic_model():
    return LogisticRegression(max_iter=1000)

def random_forest_model():
    return RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
