"""
main.py
Command-line entry point for the House Price Prediction project.
Runs the full pipeline: generate data → train → evaluate → save charts.

Usage:
    python main.py
    python main.py --predict        # Interactive price prediction
    python main.py --eda            # Run EDA only
"""

import argparse
import os
import sys

sys.path.insert(0, "src")


def run_pipeline():
    print("\n" + "="*60)
    print("  🏠 HOUSE PRICE PREDICTION — Full Pipeline")
    print("="*60)

    # Step 1: Generate dataset
    print("\n[Step 1] Generating synthetic dataset…")
    from src.generate_dataset import generate_housing_data
    os.makedirs("data", exist_ok=True)
    df = generate_housing_data()
    df.to_csv("data/houses.csv", index=False)
    print(f"  ✅ Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Price range: ₹{df['SalePrice'].min():,.0f} – ₹{df['SalePrice'].max():,.0f}")

    # Step 2: Train models
    print("\n[Step 2] Training regression models…")
    from src.train_models import train_and_compare
    results, best_name = train_and_compare()

    # Step 3: EDA & visualizations
    print("\n[Step 3] Generating EDA charts…")
    from src.eda_visualize import run_eda
    run_eda()

    print("\n" + "="*60)
    print("  ✅ Pipeline complete!")
    print(f"  Best Model: {best_name}")
    print("  Run the dashboard:  streamlit run dashboard.py")
    print("="*60 + "\n")


def interactive_predict():
    print("\n🏡 Interactive House Price Predictor")
    print("-"*40)
    import joblib
    import pandas as pd
    from src.features import add_features, get_feature_lists

    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        print("No trained model found. Run 'python main.py' first.")
        return

    pipe = joblib.load(model_path)
    NUM, CAT = get_feature_lists()

    print("Enter property details (press Enter for defaults):\n")
    def ask(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else str(default)

    neigh    = ask("Neighborhood", "Koramangala")
    ms_zone  = ask("MSZoning (RL/RM/C/FV)", "RL")
    gr_liv   = int(ask("Above-Ground Living Area (sq ft)", 1600))
    qual     = int(ask("Overall Quality (1–10)", 7))
    yr_built = int(ask("Year Built", 2005))
    full_b   = int(ask("Full Bathrooms", 2))
    garage   = float(ask("Garage Cars", 2))
    kitch    = ask("Kitchen Quality (Ex/Gd/TA/Fa)", "Gd")

    row = {
        "MSZoning": ms_zone, "LotArea": 6000, "Neighborhood": neigh,
        "BldgType": "1Fam", "HouseStyle": "2Story", "OverallQual": qual,
        "OverallCond": 5, "YearBuilt": yr_built, "YearRemodAdd": yr_built + 5,
        "Foundation": "PConc", "TotalBsmtSF": 800.0, "1stFlrSF": gr_liv // 2,
        "2ndFlrSF": gr_liv // 2, "GrLivArea": gr_liv, "FullBath": full_b,
        "HalfBath": 1, "BedroomAbvGr": 3, "TotRmsAbvGrd": 7, "Fireplaces": 1,
        "GarageCars": garage, "GarageArea": garage * 200, "PavedDrive": "Y",
        "CentralAir": "Y", "KitchenQual": kitch,
    }
    inp = pd.DataFrame([row])
    inp = add_features(inp)
    price = pipe.predict(inp[NUM + CAT])[0]
    print(f"\n  💰 Predicted Price: ₹{price:,.0f}")
    print(f"  Range (±8%):      ₹{price*0.92:,.0f} – ₹{price*1.08:,.0f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument("--predict", action="store_true", help="Run interactive predictor")
    parser.add_argument("--eda",     action="store_true", help="Run EDA only")
    args = parser.parse_args()

    if args.predict:
        interactive_predict()
    elif args.eda:
        from src.eda_visualize import run_eda
        run_eda()
    else:
        run_pipeline()
