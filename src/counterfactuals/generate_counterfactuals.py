# src/counterfactuals/generate_counterfactuals.py

import pandas as pd
import numpy as np
from counterfactuals.clinical_constraints import CLINICAL_CONSTRAINTS


def generate_counterfactual(
    model,
    X,
    subject_index,
    threshold=0.5,
    step_size=0.05
):
    """
    Generate a counterfactual for one subject.
    """

    x_orig = X.iloc[subject_index].copy()
    x_cf = x_orig.copy()

    prob_orig = model.predict_proba(
        x_orig.values.reshape(1, -1)
    )[0, 1]

    if prob_orig < threshold:
        return None  # already low risk

    for feature, rule in CLINICAL_CONSTRAINTS.items():

        if feature not in x_cf.index:
            continue

        direction = rule["direction"]
        max_change = rule["max_change"]

        for step in np.arange(step_size, max_change + step_size, step_size):

            if direction == "decrease":
                x_cf[feature] = x_orig[feature] * (1 - step)
            else:
                x_cf[feature] = x_orig[feature] * (1 + step)

            prob_new = model.predict_proba(
                x_cf.values.reshape(1, -1)
            )[0, 1]

            if prob_new < threshold:
                return {
                    "original_risk": prob_orig,
                    "counterfactual_risk": prob_new,
                    "changed_feature": feature,
                    "relative_change": step
                }

        x_cf[feature] = x_orig[feature]

    return None
