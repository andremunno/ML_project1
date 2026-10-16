import csv
import json

from src.config import DATA_DIR, METADATA_PATH


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
