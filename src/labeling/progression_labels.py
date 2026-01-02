# src/labeling/progression_labels.py

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from preprocessing.load_data import load_diadata_long_format


def compute_slope(time_values, glucose_values):
    model = LinearRegression()
    model.fit(time_values.reshape(-1, 1), glucose_values)
    return model.coef_[0]


def generate_progression_labels(
    data_files,
    output_path,
    mean_glucose_threshold=140.0,
    trend_percentile=75,
    min_samples=50
):
    """
    Generate early worsening of glucose control labels
    using chunk-based DiaData loading.
    """

    subject_series = load_diadata_long_format(data_files)

    records = []

    for pid, df in subject_series.items():

        if len(df) < min_samples:
            continue

        time_index = np.arange(len(df))
        split = len(df) // 2

        early = df.iloc[:split]
        late = df.iloc[split:]

        beta_early = compute_slope(
            time_index[:split],
            early["glucose"].values
        )

        beta_late = compute_slope(
            time_index[split:],
            late["glucose"].values
        )

        delta_beta = beta_late - beta_early
        late_mean = late["glucose"].mean()

        records.append({
            "subject_id": pid,
            "beta_early": beta_early,
            "beta_late": beta_late,
            "delta_slope": delta_beta,
            "late_mean_glucose": late_mean
        })

    label_df = pd.DataFrame(records)

    trend_threshold = np.percentile(
        label_df["delta_slope"], trend_percentile
    )

    label_df["label"] = (
        (label_df["delta_slope"] > trend_threshold) &
        (label_df["late_mean_glucose"] > mean_glucose_threshold)
    ).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    label_df.to_csv(output_path, index=False)

    print("Label generation completed.")
    print(f"Subjects: {len(label_df)}")
    print(f"Positive cases: {label_df['label'].sum()}")

    return label_df
