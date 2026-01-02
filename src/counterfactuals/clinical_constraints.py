# src/counterfactuals/clinical_constraints.py

CLINICAL_CONSTRAINTS = {
    "glucose_std": {
        "direction": "decrease",
        "max_change": 0.20  # 20%
    },
    "glucose_cv": {
        "direction": "decrease",
        "max_change": 0.20
    },
    "pct_hyperglycemia": {
        "direction": "decrease",
        "max_change": 0.30
    },
    "time_in_range": {
        "direction": "increase",
        "max_change": 0.30
    }
}
