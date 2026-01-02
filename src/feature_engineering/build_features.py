# src/feature_engineering/build_features.py

import os
from preprocessing.load_data import load_diadata_long_format
from feature_engineering.temporal_features import extract_longitudinal_features


def build_features(data_files, output_path):

    subject_series = load_diadata_long_format(data_files)
    features_df = extract_longitudinal_features(subject_series)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print("Feature extraction completed.")
    print(f"Subjects with features: {len(features_df)}")

    return features_df
