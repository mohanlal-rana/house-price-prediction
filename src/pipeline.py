"""
pipeline.py
Builds a scikit-learn ColumnTransformer pipeline for numeric + categorical features.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from features import get_feature_lists


def build_preprocessor():
    """Return a fitted-ready ColumnTransformer."""
    NUM, CAT = get_feature_lists()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM),
        ("cat", cat_pipeline, CAT)
    ], remainder="drop")

    return preprocessor
