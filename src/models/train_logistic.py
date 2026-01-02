# src/models/train_logistic.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


def train_logistic(X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model": "Logistic Regression",
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred)
    }

    return model, results
