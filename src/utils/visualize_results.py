# src/utils/visualize_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Helper
# --------------------------------------------------
def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# --------------------------------------------------
# FIGURE 1: Label Distribution
# --------------------------------------------------
def plot_label_distribution(labels_path, output_path):
    labels = pd.read_csv(labels_path)

    counts = labels["label"].value_counts().sort_index()

    plt.figure()
    plt.bar(["No Progression", "Progression"], counts.values)
    plt.ylabel("Number of Subjects")
    plt.title("Label Distribution")

    save_fig(output_path)


# --------------------------------------------------
# FIGURE 2: SHAP Feature Importance (Global)
# --------------------------------------------------
def plot_shap_importance(shap_values_path, output_path, top_k=10):
    shap_df = pd.read_csv(shap_values_path)

    importance = shap_df.abs().mean().sort_values(ascending=False)[:top_k]

    plt.figure()
    plt.barh(importance.index[::-1], importance.values[::-1])
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Global Feature Importance (SHAP)")

    save_fig(output_path)


# --------------------------------------------------
# FIGURE 3: Temporal Feature Summary
# --------------------------------------------------
def plot_temporal_features(summary_path, output_path):
    df = pd.read_csv(summary_path)

    plt.figure()
    plt.barh(df["feature"], df["mean_absolute_value"])
    plt.xlabel("Mean Absolute Value")
    plt.title("Temporal Feature Influence")

    save_fig(output_path)


# --------------------------------------------------
# FIGURE 4: Model Performance Comparison
# --------------------------------------------------
def plot_model_performance(results_path, output_path):
    results = pd.read_csv(results_path)

    x = range(len(results))

    plt.figure()
    plt.bar(x, results["auc"], label="AUC")
    plt.bar(x, results["f1"], bottom=0, alpha=0.7, label="F1")

    plt.xticks(x, results["model"], rotation=15)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()

    save_fig(output_path)


# --------------------------------------------------
# FIGURE 5: Counterfactual Effect
# --------------------------------------------------
def plot_counterfactuals(counterfactuals_path, output_path):
    df = pd.read_csv(counterfactuals_path)

    if df.empty:
        print("No counterfactuals found — skipping plot.")
        return

    plt.figure()
    plt.hist(
        df["relative_change"],
        bins=10
    )
    plt.xlabel("Relative Feature Change")
    plt.ylabel("Number of Subjects")
    plt.title("Counterfactual Intervention Magnitude")

    save_fig(output_path)


# --------------------------------------------------
# MASTER FUNCTION
# --------------------------------------------------
def generate_all_figures():
    plot_label_distribution(
        labels_path="data/processed/labels.csv",
        output_path="results/figures/fig1_label_distribution.png"
    )

    plot_shap_importance(
        shap_values_path="results/figures/XGBoost_shap_values.csv",
        output_path="results/figures/fig2_shap_importance.png"
    )

    plot_temporal_features(
        summary_path="results/figures/temporal_feature_summary.csv",
        output_path="results/figures/fig3_temporal_features.png"
    )

    plot_model_performance(
        results_path="results/tables/model_results.csv",
        output_path="results/figures/fig4_model_performance.png"
    )

    plot_counterfactuals(
        counterfactuals_path="results/tables/counterfactuals.csv",
        output_path="results/figures/fig5_counterfactuals.png"
    )

    print("All figures generated successfully.")
