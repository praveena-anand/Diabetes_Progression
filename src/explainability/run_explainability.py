# src/explainability/run_explainability.py

import os
import pandas as pd

from models.prepare_dataset import prepare_dataset
from models.train_logistic import train_logistic
from models.train_xgboost import train_xgboost
from explainability.shap_analysis import run_shap_analysis
from explainability.temporal_importance import summarize_temporal_features


def run_explainability(features_path, labels_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_dataset(
        features_path, labels_path
    )

    # Train models (same as Step 3)
    log_model, _ = train_logistic(X_train, X_test, y_train, y_test)
    xgb_model, _ = train_xgboost(X_train, X_test, y_train, y_test, threshold=0.3)

    # SHAP analysis
    run_shap_analysis(
        log_model, X_train, X_test,
        model_name="Logistic_Regression",
        output_dir=output_dir
    )

    run_shap_analysis(
        xgb_model, X_train, X_test,
        model_name="XGBoost",
        output_dir=output_dir
    )

    # Temporal summary
    features_df = pd.read_csv(features_path)
    temporal_summary = summarize_temporal_features(features_df)

    temporal_summary.to_csv(
        f"{output_dir}/temporal_feature_summary.csv",
        index=False
    )

    print("Explainability analysis completed.")
