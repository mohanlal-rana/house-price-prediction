"""
eda_visualize.py
Performs EDA and saves all charts to outputs/images/.
Run once to generate all images for GitHub.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from features import add_features

sns.set_theme(style="whitegrid", palette="muted")
IMG_DIR = os.path.join(os.path.dirname(__file__), "../images")
os.makedirs(IMG_DIR, exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, name), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")


def run_eda():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/houses.csv"))
    df = add_features(df)

    # ── 1. SalePrice distribution ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["SalePrice"] / 1e6, bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Sale Price Distribution")
    axes[0].set_xlabel("Price (₹ Millions)")
    axes[0].set_ylabel("Count")
    axes[1].hist(np.log1p(df["SalePrice"]), bins=40, color="#DD8452", edgecolor="white")
    axes[1].set_title("Log-transformed Sale Price")
    axes[1].set_xlabel("log(1 + SalePrice)")
    axes[1].set_ylabel("Count")
    savefig("01_price_distribution.png")

    # ── 2. Correlation heatmap (numeric) ───────────────────────────────────────
    num_cols = ["SalePrice", "GrLivArea", "TotalBsmtSF", "LotArea",
                "OverallQual", "GarageCars", "FullBath", "Age",
                "TotalSF", "BathsTotal", "Fireplaces"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    savefig("02_correlation_heatmap.png")

    # ── 3. Price vs GrLivArea ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df["GrLivArea"], df["SalePrice"] / 1e6,
                    c=df["OverallQual"], cmap="viridis", alpha=0.5, s=20)
    plt.colorbar(sc, ax=ax, label="Overall Quality")
    ax.set_xlabel("Above-Ground Living Area (sq ft)")
    ax.set_ylabel("Sale Price (₹ Millions)")
    ax.set_title("Price vs Living Area (colored by Quality)")
    savefig("03_price_vs_area.png")

    # ── 4. Price by Neighborhood ───────────────────────────────────────────────
    med_price = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    med_price.div(1e6).plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(med_price)))
    ax.set_title("Median Sale Price by Neighborhood")
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("Median Price (₹ Millions)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("₹%.1fM"))
    ax.tick_params(axis="x", rotation=30)
    savefig("04_price_by_neighborhood.png")

    # ── 5. Overall Quality vs Price (box) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df["OverallQual"].unique())
    sns.boxplot(data=df, x="OverallQual", y="SalePrice", order=order, ax=ax, palette="Blues")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1e6:.1f}M"))
    ax.set_title("Sale Price Distribution by Overall Quality")
    ax.set_xlabel("Overall Quality (1–10)")
    ax.set_ylabel("Sale Price")
    savefig("05_price_by_quality.png")

    # ── 6. Age vs Price ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Age"], df["SalePrice"] / 1e6, alpha=0.3, s=15, color="#55a868")
    z = np.polyfit(df["Age"], df["SalePrice"] / 1e6, 1)
    p = np.poly1d(z)
    xs = np.linspace(df["Age"].min(), df["Age"].max(), 200)
    ax.plot(xs, p(xs), "r--", lw=2, label="Trend")
    ax.set_xlabel("House Age (years)")
    ax.set_ylabel("Sale Price (₹ Millions)")
    ax.set_title("House Age vs Sale Price")
    ax.legend()
    savefig("06_age_vs_price.png")

    # ── 7. Model Comparison Bar Chart (loaded from results.json) ───────────────
    import json
    results_path = os.path.join(os.path.dirname(__file__), "../models/results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        names = list(results.keys())
        rmses = [results[n]["test_rmse"] / 1e6 for n in names]
        r2s   = [results[n]["test_r2"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = sns.color_palette("muted", len(names))
        axes[0].barh(names, rmses, color=colors)
        axes[0].set_xlabel("Test RMSE (₹ Millions)")
        axes[0].set_title("Model RMSE Comparison (lower is better)")
        axes[1].barh(names, r2s, color=colors)
        axes[1].set_xlabel("Test R² Score")
        axes[1].set_title("Model R² Comparison (higher is better)")
        savefig("07_model_comparison.png")

    # ── 8. Actual vs Predicted ──────────────────────────────────────────────────
    import joblib
    from sklearn.model_selection import train_test_split
    from features import get_feature_lists

    model_path = os.path.join(os.path.dirname(__file__), "../models/best_model.joblib")
    if os.path.exists(model_path):
        pipe = joblib.load(model_path)
        NUM, CAT = get_feature_lists()
        X = df[NUM + CAT]
        y = df["SalePrice"]
        _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        pred = pipe.predict(Xte)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(yte / 1e6, pred / 1e6, alpha=0.3, s=18, color="#4C72B0")
        lims = [min(yte.min(), pred.min()) / 1e6, max(yte.max(), pred.max()) / 1e6]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
        ax.set_xlabel("Actual Price (₹ Millions)")
        ax.set_ylabel("Predicted Price (₹ Millions)")
        ax.set_title("Actual vs Predicted Sale Price")
        ax.legend()
        savefig("08_actual_vs_predicted.png")

        # Residual plot
        residuals = yte - pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(pred / 1e6, residuals / 1e6, alpha=0.3, s=15, color="#DD8452")
        axes[0].axhline(0, color="red", lw=2, linestyle="--")
        axes[0].set_xlabel("Predicted Price (₹ Millions)")
        axes[0].set_ylabel("Residual (₹ Millions)")
        axes[0].set_title("Residual Plot")
        axes[1].hist(residuals / 1e6, bins=40, color="#55a868", edgecolor="white")
        axes[1].set_xlabel("Residual (₹ Millions)")
        axes[1].set_title("Residual Distribution")
        savefig("09_residual_analysis.png")

    print("\n✅ All EDA charts saved to images/")


if __name__ == "__main__":
    run_eda()
