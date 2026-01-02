# src/models/train_xgboost.py

import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score


def train_xgboost(X_train, X_test, y_train, y_test, threshold=0.3):

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuning (IMPORTANT)
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "model": f"XGBoost (thr={threshold})",
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred)
    }

    return model, results
