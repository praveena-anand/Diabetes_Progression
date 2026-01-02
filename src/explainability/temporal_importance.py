# src/explainability/temporal_importance.py

import pandas as pd


def summarize_temporal_features(features_df):
    """
    Extract and rank temporal / progression-related features.
    """

    temporal_features = [
        "global_slope",
        "glucose_std",
        "glucose_cv",
        "pct_hyperglycemia",
        "time_in_range",
        "mean_diff"
    ]

    available = [f for f in temporal_features if f in features_df.columns]

    summary = (
        features_df[available]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    summary.columns = ["feature", "mean_absolute_value"]

    return summary
