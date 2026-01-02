# src/models/train_random_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score


def train_random_forest(X_train, X_test, y_train, y_test, threshold=0.3):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuning (IMPORTANT)
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "model": f"Random Forest (thr={threshold})",
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred)
    }

    return model, results
