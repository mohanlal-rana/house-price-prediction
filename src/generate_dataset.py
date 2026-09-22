"""
generate_dataset.py
Generates a realistic synthetic housing dataset (1500 rows) 
mimicking real estate features. Used as a public/open dataset substitute.
"""

import numpy as np
import pandas as pd
import os

def generate_housing_data(n=1500, seed=42):
    np.random.seed(seed)

    neighborhoods = ["Saket", "Bandra", "Koramangala", "Andheri", "Whitefield",
                     "Powai", "Indiranagar", "Malad", "Sector17", "Viman Nagar"]
    house_styles  = ["1Story", "2Story", "SplitLevel"]
    bldg_types    = ["1Fam", "Duplex", "Apartment"]
    zones         = ["RL", "RM", "C", "FV"]
    kitchens      = ["Ex", "Gd", "TA", "Fa"]
    foundations   = ["PConc", "CBlock", "BrkTil"]

    neigh_idx = np.random.randint(0, len(neighborhoods), n)
    # Base price per neighborhood
    neigh_base = [5e6, 9e6, 7e6, 6e6, 8e6, 7.5e6, 8.5e6, 5.5e6, 6e6, 7e6]

    overall_qual = np.random.randint(3, 11, n)          # 3–10
    overall_cond = np.random.randint(3, 10, n)          # 3–9
    year_built   = np.random.randint(1970, 2023, n)
    year_remod   = year_built + np.random.randint(0, 20, n)
    year_remod   = np.clip(year_remod, year_built, 2024)
    lot_area     = np.random.normal(7000, 2000, n).clip(2000, 20000).astype(int)
    gr_liv_area  = np.random.normal(1600, 500, n).clip(600, 4000).astype(int)
    total_bsmt   = (gr_liv_area * np.random.uniform(0.4, 0.8, n)).astype(int)
    first_flr    = (gr_liv_area * np.random.uniform(0.5, 1.0, n)).astype(int)
    second_flr   = gr_liv_area - first_flr
    full_bath    = np.random.randint(1, 4, n)
    half_bath    = np.random.randint(0, 2, n)
    bedrooms     = np.random.randint(2, 6, n)
    tot_rooms    = bedrooms + full_bath + np.random.randint(1, 3, n)
    fireplaces   = np.random.randint(0, 3, n)
    garage_cars  = np.random.randint(0, 4, n)
    garage_area  = (garage_cars * 200 + np.random.normal(0, 50, n)).clip(0, 900).astype(int)
    paved_drive  = np.random.choice(["Y", "N"], n, p=[0.8, 0.2])
    central_air  = np.random.choice(["Y", "N"], n, p=[0.85, 0.15])
    ms_zoning    = np.random.choice(zones, n, p=[0.6, 0.2, 0.1, 0.1])
    house_style  = np.random.choice(house_styles, n, p=[0.5, 0.35, 0.15])
    bldg_type    = np.random.choice(bldg_types, n, p=[0.65, 0.2, 0.15])
    kitchen_qual = np.random.choice(kitchens, n, p=[0.2, 0.35, 0.35, 0.1])
    foundation   = np.random.choice(foundations, n, p=[0.5, 0.35, 0.15])

    # Price formula (realistic drivers)
    qual_map    = {"Ex":1.25, "Gd":1.1, "TA":1.0, "Fa":0.85}
    kit_factor  = np.array([qual_map[k] for k in kitchen_qual])
    ca_factor   = np.where(central_air == "Y", 1.05, 0.95)
    age_factor  = 1 - (2024 - year_built) * 0.003
    remod_bonus = np.where(year_remod > year_built + 5, 1.04, 1.0)

    price = (
        np.array([neigh_base[i] for i in neigh_idx])
        + gr_liv_area * 1800
        + total_bsmt * 600
        + overall_qual * 150000
        + garage_cars * 80000
        + fireplaces * 50000
        + full_bath * 120000
        - (2024 - year_built) * 8000
    ) * kit_factor * ca_factor * age_factor * remod_bonus

    noise = np.random.normal(0, price * 0.05)
    price = (price + noise).clip(1_000_000).astype(int)

    df = pd.DataFrame({
        "Id": range(1, n+1),
        "MSZoning": ms_zoning,
        "LotArea": lot_area,
        "Neighborhood": [neighborhoods[i] for i in neigh_idx],
        "BldgType": bldg_type,
        "HouseStyle": house_style,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "Foundation": foundation,
        "TotalBsmtSF": total_bsmt,
        "1stFlrSF": first_flr,
        "2ndFlrSF": second_flr,
        "GrLivArea": gr_liv_area,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": tot_rooms,
        "Fireplaces": fireplaces,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "PavedDrive": paved_drive,
        "CentralAir": central_air,
        "KitchenQual": kitchen_qual,
        "SalePrice": price
    })

    # Introduce ~5% missing values in some columns (realistic)
    for col in ["TotalBsmtSF", "GarageCars", "GarageArea"]:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_housing_data()
    df.to_csv("data/houses.csv", index=False)
    print(f"Dataset saved: {df.shape}")
    print(df["SalePrice"].describe())
