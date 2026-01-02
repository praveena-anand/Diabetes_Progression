# src/explainability/shap_analysis.py

import shap
import pandas as pd
import matplotlib.pyplot as plt


def run_shap_analysis(model, X_train, X_test, model_name, output_dir):
    """
    Run SHAP explainability for tree-based or linear models.
    """

    # Use appropriate explainer
    if model_name.lower().startswith("logistic"):
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    # Convert to DataFrame for saving
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_test.columns
    )

    # Save SHAP values
    shap_df.to_csv(
        f"{output_dir}/{model_name}_shap_values.csv",
        index=False
    )

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title(f"SHAP Summary – {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_shap_summary.png")
    plt.close()

    print(f"SHAP analysis completed for {model_name}")
