# src/counterfactuals/run_counterfactuals.py

import os
import pandas as pd
from models.prepare_dataset import prepare_dataset
from models.train_logistic import train_logistic
from counterfactuals.generate_counterfactuals import generate_counterfactual


def run_counterfactuals(
    features_path,
    labels_path,
    output_path,
    max_subjects=50
):

    X_train, X_test, y_train, y_test = prepare_dataset(
        features_path, labels_path
    )

    model, _ = train_logistic(X_train, X_test, y_train, y_test)

    results = []

    for idx in range(min(max_subjects, len(X_test))):

        cf = generate_counterfactual(
            model, X_test, idx
        )

        if cf is not None:
            cf["subject_index"] = idx
            results.append(cf)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print("Counterfactual generation completed.")
    print(f"Valid counterfactuals found: {len(results)}")
