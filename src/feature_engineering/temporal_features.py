# src/feature_engineering/temporal_features.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_slope(y):
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]


def extract_longitudinal_features(subject_series):
    """
    subject_series: dict {subject_id: DataFrame(ts, glucose)}
    Returns DataFrame with one row per subject.
    """

    feature_rows = []

    for pid, df in subject_series.items():

        if len(df) < 50:
            continue

        glucose = df["glucose"].values
        n = len(glucose)
        split = n // 2

        early = glucose[:split]
        late = glucose[split:]

        row = {
            "subject_id": pid,

            # Global statistics
            "glucose_mean": np.mean(glucose),
            "glucose_std": np.std(glucose),
            "glucose_min": np.min(glucose),
            "glucose_max": np.max(glucose),
            "glucose_cv": np.std(glucose) / (np.mean(glucose) + 1e-6),

            # Trend features
            "global_slope": compute_slope(glucose),
            "early_slope": compute_slope(early),
            "late_slope": compute_slope(late),
            "slope_change": compute_slope(late) - compute_slope(early),

            # Window distribution
            "early_mean": np.mean(early),
            "late_mean": np.mean(late),
            "early_std": np.std(early),
            "late_std": np.std(late),
            "mean_diff": np.mean(late) - np.mean(early),

            # Risk exposure
            "pct_hyperglycemia": np.mean(glucose > 180),
            "pct_hypoglycemia": np.mean(glucose < 70),
            "time_in_range": np.mean((glucose >= 70) & (glucose <= 180))
        }

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)
