# src/models/run_training.py

import pandas as pd
from models.prepare_dataset import prepare_dataset
from models.train_logistic import train_logistic
from models.train_random_forest import train_random_forest
from models.train_xgboost import train_xgboost


def run_training(features_path, labels_path, output_path):

    X_train, X_test, y_train, y_test = prepare_dataset(
        features_path, labels_path
    )

    results = []

    # Logistic Regression (baseline threshold = 0.5)
    _, res = train_logistic(X_train, X_test, y_train, y_test)
    results.append(res)

    # Random Forest (threshold tuned)
    _, res = train_random_forest(
        X_train, X_test, y_train, y_test, threshold=0.3
    )
    results.append(res)

    # XGBoost (threshold tuned)
    _, res = train_xgboost(
        X_train, X_test, y_train, y_test, threshold=0.3
    )
    results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print("\nModel training completed.")
    print(results_df)

    return results_df
