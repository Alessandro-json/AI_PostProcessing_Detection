"""
compare_freq.py

Builds the comparison table between all models in the project,
including the frequency branch, and plots accuracy charts.

Usage in notebook:
    %run src/compare_freq.py
    # oppure
    from compare_freq import build_comparison_table, plot_comparison
    df = build_comparison_table(results)
    plot_comparison(df)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


# ---------------------------------------------------------------------------
# Default results dictionary
# ---------------------------------------------------------------------------
# Maps a human-readable model name to the metrics.json path produced by
# the corresponding evaluate_*.py script.
# Add or remove entries to match what you have actually trained.

DEFAULT_RESULTS = {
    # RGB baselines (from evaluate_RGB.py)
    "RGB fake-only":           "results/rgb_fake/metrics.json",
    "RGB transform-only":      "results/rgb_transform/metrics.json",
    "RGB multitask 1-1":       "results/rgb_multitask_1_1/metrics.json",
    "RGB multitask 1-2":       "results/rgb_multitask_1_2/metrics.json",
    "RGB multitask 2-1":       "results/rgb_multitask_2_1/metrics.json",
    "RGB multitask learned":   "results/rgb_multitask_learned_weights/metrics.json",

    # Depth + frequency model (from evaluation_depth_frequency.py)
    "RGB+Depth+Freq 1-1":      "results/depth_frequency_1_1/metrics.json",
    "RGB+Depth+Freq learned":  "results/depth_frequency_learned/metrics.json",

    # Frequency branch — your contribution (from evaluate_freq.py)
    "Freq 1-1":                "results/freq_1_1/metrics.json",
    "Freq 1-2":                "results/freq_1_2/metrics.json",
    "Freq 2-1":                "results/freq_2_1/metrics.json",
    "Freq learned cosine":     "results/freq_learned_cosine/metrics.json",

    # ViT baselines (from evaluate_vit_RGB.py)
    "ViT RGB fake-only":       "results/vit_rgb_fake/metrics.json",
    "ViT RGB multitask 1-1":   "results/vit_rgb_multitask_1_1/metrics.json",
    "ViT RGB multitask learned":"results/vit_rgb_multitask_learned/metrics.json",

    # ViT + depth + frequency (from evaluate_vit_depth_frequency.py)
    "ViT+Depth+Freq 1-1":      "results/vit_depth_frequency_1_1/metrics.json",
    "ViT+Depth+Freq learned":  "results/vit_depth_frequency_learned/metrics.json",
}


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict = None) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dictionary of metrics.json paths.

    Args:
        results: dict mapping model_name -> metrics.json path.
                 If None, uses DEFAULT_RESULTS.

    Returns:
        DataFrame with columns:
            model, fake_accuracy, fake_f1_macro,
            transform_accuracy, transform_f1_macro,
            fake_acc_original, fake_acc_transfer, fake_acc_redigital
    """

    if results is None:
        results = DEFAULT_RESULTS

    rows = []

    for model_name, metrics_path in results.items():

        metrics_path = Path(metrics_path)

        if not metrics_path.exists():
            print(f"[SKIP] Missing: {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        by_transform = metrics.get("fake_accuracy_by_transform", {})

        rows.append({
            "model":              model_name,
            "fake_accuracy":      metrics.get("fake_accuracy"),
            "fake_f1_macro":      metrics.get("fake_f1_macro"),
            "transform_accuracy": metrics.get("transform_accuracy"),
            "transform_f1_macro": metrics.get("transform_f1_macro"),
            "fake_acc_original":  by_transform.get("original"),
            "fake_acc_transfer":  by_transform.get("transfer"),
            "fake_acc_redigital": by_transform.get("redigital"),
        })

    if not rows:
        print("No results found. Run the evaluation scripts first.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by fake_accuracy descending so the best model is at the top.
    df = df.sort_values(
        by=["fake_accuracy", "transform_accuracy"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame):
    """
    Plot two bar charts from the comparison DataFrame:
        1. Global fake and transform accuracy across all models.
        2. Real/fake accuracy broken down by transformation type.

    Args:
        df: DataFrame returned by build_comparison_table().
    """

    if df.empty:
        print("No data to plot.")
        return

    plot_df = df.set_index("model")

    # --- Chart 1: Global accuracy ---
    fig, ax = plt.subplots(figsize=(max(12, len(df) * 1.2), 5))

    plot_df[["fake_accuracy", "transform_accuracy"]].plot(
        kind="bar",
        ax=ax,
        color=["steelblue", "coral"],
        edgecolor="white",
        width=0.7,
    )

    ax.set_title("Global Accuracy — All Models", fontsize=13, pad=12)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.legend(["Fake accuracy", "Transform accuracy"])
    ax.grid(axis="y", alpha=0.4)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")

    plt.tight_layout()
    plt.savefig("results/comparison_global_accuracy.png", dpi=200)
    plt.show()
    print("Saved: results/comparison_global_accuracy.png")

    # --- Chart 2: Real/fake accuracy by transformation ---
    transform_cols = ["fake_acc_original", "fake_acc_transfer", "fake_acc_redigital"]

    # Only plot models that have per-transformation data.
    available = plot_df[transform_cols].dropna(how="all")

    if available.empty:
        print("No per-transformation data available yet.")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(available) * 1.2), 5))

    available[transform_cols].plot(
        kind="bar",
        ax=ax,
        color=["#4c72b0", "#55a868", "#c44e52"],
        edgecolor="white",
        width=0.7,
    )

    ax.set_title("Real/Fake Accuracy by Transformation Type", fontsize=13, pad=12)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.legend(["Original", "Transfer", "Re-digital"])
    ax.grid(axis="y", alpha=0.4)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("results/comparison_by_transform.png", dpi=200)
    plt.show()
    print("Saved: results/comparison_by_transform.png")


# ---------------------------------------------------------------------------
# Main (usable both as script and as imported module)
# ---------------------------------------------------------------------------

def main():
    """
    Build and display the full comparison table, then plot accuracy charts.

    Can be run directly:
        python src/compare_freq.py

    Or imported in a notebook cell:
        from compare_freq import build_comparison_table, plot_comparison
    """

    print("Building comparison table...\n")

    df = build_comparison_table()

    if df.empty:
        return

    # Format numeric columns for readability.
    numeric_cols = [
        "fake_accuracy", "fake_f1_macro",
        "transform_accuracy", "transform_f1_macro",
        "fake_acc_original", "fake_acc_transfer", "fake_acc_redigital",
    ]

    df_display = df.copy()
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "-"
            )

    display(df_display)

    # Save raw table to CSV for the report.
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/comparison_table.csv", index=False)
    print("\nSaved: results/comparison_table.csv")

    plot_comparison(df)


if __name__ == "__main__":
    main()
