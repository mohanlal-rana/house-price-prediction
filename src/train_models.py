"""
train_models.py
Trains multiple regression models, compares their performance,
and saves the best model to the models/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import add_features, get_feature_lists
from pipeline import build_preprocessor


def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/houses.csv"))
    df = add_features(df)
    NUM, CAT = get_feature_lists()
    feature_cols = NUM + CAT
    X = df[feature_cols]
    y = df["SalePrice"]
    return X, y


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_and_compare():
    X, y = load_data()
    pre = build_preprocessor()

    models = {
        "Linear Regression": Pipeline([("pre", pre), ("model", LinearRegression())]),
        "Ridge Regression":  Pipeline([("pre", build_preprocessor()), ("model", Ridge(alpha=10.0))]),
        "Decision Tree":     Pipeline([("pre", build_preprocessor()), ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42))]),
        "Random Forest":     Pipeline([("pre", build_preprocessor()), ("model", RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("pre", build_preprocessor()), ("model", GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42))]),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*70)
    print(f"{'Model':<22} {'CV RMSE':>14} {'CV R²':>10}")
    print("="*70)

    results = {}
    for name, pipe in models.items():
        cv_rmse = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        cv_r2   =  cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        results[name] = {"cv_rmse_mean": cv_rmse.mean(), "cv_rmse_std": cv_rmse.std(), "cv_r2_mean": cv_r2.mean()}
        print(f"{name:<22} {cv_rmse.mean():>10,.0f} ± {cv_rmse.std():>6,.0f}   {cv_r2.mean():>8.4f}")

    print("="*70)

    # --- Final train/test split evaluation ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Hold-out Test Set Metrics ---")
    print(f"{'Model':<22} {'MAE':>12} {'RMSE':>12} {'R²':>8}")
    print("-"*60)

    best_name, best_pipe, best_rmse = None, None, float("inf")
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        mae  = mean_absolute_error(yte, pred)
        rms  = rmse(yte, pred)
        r2   = r2_score(yte, pred)
        results[name]["test_mae"]  = mae
        results[name]["test_rmse"] = rms
        results[name]["test_r2"]   = r2
        print(f"{name:<22} ₹{mae:>10,.0f}   ₹{rms:>10,.0f}   {r2:>6.4f}")
        if rms < best_rmse:
            best_rmse, best_name, best_pipe = rms, name, pipe

    print("-"*60)
    print(f"\n✅ Best Model: {best_name}  (Test RMSE = ₹{best_rmse:,.0f})")

    # Save best model
    os.makedirs(os.path.join(os.path.dirname(__file__), "../models"), exist_ok=True)
    joblib.dump(best_pipe, os.path.join(os.path.dirname(__file__), "../models/best_model.joblib"))
    joblib.dump(models,    os.path.join(os.path.dirname(__file__), "../models/all_models.joblib"))
    print("💾 Models saved to models/")

    # Save results for dashboard
    import json
    with open(os.path.join(os.path.dirname(__file__), "../models/results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, best_name


if __name__ == "__main__":
    train_and_compare()
