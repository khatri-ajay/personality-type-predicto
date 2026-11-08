"""Shared helpers: data loading/cleaning, the preprocessing pipeline, MLflow location."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"

# One absolute path, built from this file's location, so the notebooks, the
# MLflow UI and the Streamlit app all point at the same database
# (four slashes on Linux/macOS: "sqlite:///" + "/abs/path").
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "personality-type"

DEMOGRAPHICS = ["age", "gender", "hand"]
CATEGORICAL = ["gender", "hand"]


def load_clean_data(path=DATA_PATH):
    """Load data.csv and remove rows that cannot be real answers."""
    df = pd.read_csv(path)
    item_cols = [c for c in df.columns if c not in DEMOGRAPHICS + ["target"]]
    all_zero = (df[item_cols] == 0).all(axis=1)       # item scale is 1-5; 0 is junk
    bad_age = ~df["age"].between(13, 100)              # e.g. 999999999 or birth years
    return df.loc[~(all_zero | bad_age)].reset_index(drop=True)


def split_xy(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def build_preprocessor(X):
    item_cols = [c for c in X.columns if c not in DEMOGRAPHICS]
    numeric = item_cols + ["age"]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])