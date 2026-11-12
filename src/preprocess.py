"""Shared data loading and preprocessing for the Airline Satisfaction project.

Everything that learns from data (imputation values, scaling, encoding) lives
inside an sklearn Pipeline, so it is fitted on the training fold only and
never sees the validation fold (no leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = "satisfaction"
ID_COL = "id"

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
DELAY_COLS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
NUMERIC_COLS = ["Age", "Flight Distance"] + DELAY_COLS
BINARY_COLS = ["Gender", "Customer Type", "Type of Travel"]
NOMINAL_COLS = ["Class"]
CATEGORICAL_COLS = BINARY_COLS + NOMINAL_COLS

# Fixed binary mappings (no data is learned, so no leakage risk)
BINARY_MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Customer Type": {"disloyal Customer": 0, "Loyal Customer": 1},
    "Type of Travel": {"Personal Travel": 0, "Business travel": 1},
}


def load_data(data_dir: Path = DATA_DIR):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train[TARGET] = train[TARGET].astype(int)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning that uses no statistics from the data.

    * Arrival delay is missing for ~0.04% of rows. Departure and arrival delay
      are strongly correlated (r = 0.84), and a flight that left on time
      usually lands on time, so we fill the gap with the departure delay of
      the same row and keep a flag so the model can see it was imputed.
    * Rating 0 means "not applicable" (passenger did not use the service).
      It is kept as its own value; a count of such answers is added.
    """
    df = df.copy()
    df["arrival_delay_missing"] = df["Arrival Delay in Minutes"].isna().astype(int)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Departure Delay in Minutes"]
    )
    df["n_not_applicable"] = (df[RATING_COLS] == 0).sum(axis=1)
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping).astype(int)
    return df


def feature_columns():
    return NUMERIC_COLS + RATING_COLS + BINARY_COLS + NOMINAL_COLS + [
        "arrival_delay_missing", "n_not_applicable"
    ]


def linear_preprocessor() -> ColumnTransformer:
    """For linear models: log-transform skewed delays, scale, one-hot Class."""
    log_scale = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("delays", log_scale, DELAY_COLS),
        ("num", StandardScaler(), ["Age", "Flight Distance"] + RATING_COLS + ["n_not_applicable"]),
        ("class", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLS),
        ("bin", "passthrough", BINARY_COLS + ["arrival_delay_missing"]),
    ])


def tree_frame(df: pd.DataFrame) -> pd.DataFrame:
    """For tree models: no scaling needed; Class as an ordered integer
    (Eco < Eco Plus < Business), which matches the service level."""
    X = df[feature_columns()].copy()
    X["Class"] = X["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2}).astype(int)
    return X


def get_folds(y, n_splits: int = 5, seed: int = 42):
    """The one validation scheme used by every model: 5-fold stratified CV.
    Deterministic, so all notebooks get exactly the same folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
