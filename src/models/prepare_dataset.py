# src/models/prepare_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset(
    features_path,
    labels_path,
    test_size=0.2,
    random_state=42
):
    """
    Merge features and labels, remove leaky features,
    and create train-test split.
    """

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    # --------------------------------------------------
    # Merge on subject_id
    # --------------------------------------------------
    data = features.merge(
        labels[["subject_id", "label"]],
        on="subject_id",
        how="inner"
    )

    # --------------------------------------------------
    # Separate X and y
    # --------------------------------------------------
    X = data.drop(columns=["subject_id", "label"])
    y = data["label"]

    # --------------------------------------------------
    # 🚨 REMOVE LABEL-LEAKY FEATURES (CRITICAL)
    # --------------------------------------------------
    LEAKY_FEATURES = [
        "early_slope",
        "late_slope",
        "slope_change",
        "late_mean"
    ]

    X = X.drop(columns=LEAKY_FEATURES, errors="ignore")

    # --------------------------------------------------
    # Train-test split (stratified)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # --------------------------------------------------
    # Debug info (useful, reviewer-safe)
    # --------------------------------------------------
    print(f"Total samples: {len(data)}")
    print(f"Features used for training: {X.shape[1]}")
    print(f"Positive class ratio: {y.mean():.3f}")

    return X_train, X_test, y_train, y_test
