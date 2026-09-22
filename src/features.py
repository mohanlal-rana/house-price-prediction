"""
features.py
Feature engineering for house price prediction.
Adds derived features: Age, RemodAge, BathsTotal, RoomsPerArea, GarageQualCars.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the housing dataframe."""
    df = df.copy()

    # Age of the house at time of model fitting (2024 baseline)
    df["Age"] = 2024 - df["YearBuilt"]
    df["Age"] = df["Age"].clip(0, 100)

    # How recently was it remodeled?
    df["RemodAge"] = 2024 - df["YearRemodAdd"]
    df["RemodAge"] = df["RemodAge"].clip(0, 100)

    # Remodel bonus: was it remodeled after construction?
    df["WasRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)

    # Total bathrooms (half baths count as 0.5)
    df["BathsTotal"] = df["FullBath"] + 0.5 * df["HalfBath"]

    # Room density (rooms per unit area)
    df["RoomsPerArea"] = df["TotRmsAbvGrd"] / (df["GrLivArea"].replace(0, 1))

    # Garage quality composite
    df["GarageQualCars"] = df["GarageCars"].fillna(0) * (
        df["GarageArea"].fillna(0) / 200
    )

    # Price per sqft helper (for EDA only; won't be a training feature)
    # We keep it separate for visualization
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0) +
        df["1stFlrSF"] +
        df["2ndFlrSF"]
    )

    return df


def get_feature_lists():
    """Return numeric and categorical feature column lists for the pipeline."""
    NUM = [
        "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath",
        "HalfBath", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea",
        "Age", "RemodAge", "BathsTotal", "RoomsPerArea", "GarageQualCars",
        "WasRemodeled", "TotalSF"
    ]
    CAT = [
        "MSZoning", "Neighborhood", "BldgType", "HouseStyle",
        "KitchenQual", "Foundation", "PavedDrive", "CentralAir"
    ]
    return NUM, CAT
