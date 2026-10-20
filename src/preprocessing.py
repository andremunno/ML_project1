"""Data loading, feature selection and the full preprocessing pipeline (Andrea).

Pipeline: select_features -> clean (missing) -> decode (brfss) -> encode (encoding) -> standardize.
Fit on training rows only, then apply the same state to train, validation and test.
"""

import csv
import json

import numpy as np

from helpers import load_csv_data
from src.brfss import decode
from src.config import DATA_DIR, METADATA_PATH
from src.encoding import apply_encode, fit_encode
from src.missing import clean


def load_data(data_path=DATA_DIR):
    """load_csv_data, cached in data_path/cache.npz after the first (slow) call."""
    cache = data_path / "cache.npz"
    if not cache.exists():
        np.savez(cache, *load_csv_data(data_path))
    data = np.load(cache)
    return tuple(data[f"arr_{i}"] for i in range(5))


def load_feature_names(data_path=DATA_DIR):
    """Column names of x_train.csv, without the Id column (same order as load_csv_data)."""
    with open(data_path / "x_train.csv") as f:
        return f.readline().strip().split(",")[1:]


def load_metadata(path=METADATA_PATH):
    """Metadata as a dict {variable: row}, with the list columns already parsed."""
    meta = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("na_values", "zero_values", "recode"):
                row[key] = json.loads(row[key])
            row["scale"] = float(row["scale"])
            meta[row["variable"]] = row
    return meta


def select_features(meta, feature_names, max_group):
    """Indices of the features with use == "include" and group <= max_group."""
    return [
        j
        for j, name in enumerate(feature_names)
        if meta[name]["use"] == "include"
        and meta[name]["group"]
        and int(meta[name]["group"]) <= max_group
    ]


def _clean_and_decode(X_raw, meta, names, cols):
    selected = [names[j] for j in cols]
    X = clean(X_raw[:, cols], meta, selected)
    return decode(X, meta, selected)


def fit_preprocess(X_raw, meta, names, max_group, min_frequency=50):
    """Learn the preprocessing on the training rows.

    Returns:
        state: dict used by apply_preprocess
    """
    cols = select_features(meta, names, max_group)
    X, meta_dec, names_dec = _clean_and_decode(X_raw, meta, names, cols)
    encoder = fit_encode(X, meta_dec, names_dec, min_frequency)

    # Standardize every encoded column, 0/1 columns included
    X_enc, _ = apply_encode(X, encoder)
    std = X_enc.std(axis=0)
    std[std == 0] = 1.0
    return {"cols": cols, "encoder": encoder, "mean": X_enc.mean(axis=0), "std": std}


def apply_preprocess(X_raw, meta, names, state):
    """Apply the learned preprocessing to any rows (train, validation or test).

    Returns:
        X: float32 numpy array (N, D'), without NaNs and without bias column
        names: list of the D' column names
    """
    X, _, _ = _clean_and_decode(X_raw, meta, names, state["cols"])
    X_enc, out_names = apply_encode(X, state["encoder"])
    return (X_enc - state["mean"]) / state["std"], out_names
