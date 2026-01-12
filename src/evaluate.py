from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate(model, X, y):
    y_pred = model.predict(X)

    return {
        "Accuracy": round(accuracy_score(y, y_pred), 3),
        "Precision": round(precision_score(y, y_pred), 3),
        "Recall": round(recall_score(y, y_pred), 3),
        "Confusion Matrix": confusion_matrix(y, y_pred)
    }
