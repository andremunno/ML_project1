"""Metadata-driven cleaning and median imputation (Lorenzo)."""

import numpy as np


def clean(X, meta, names):
    """Apply the metadata recodings, column by column.

    na_values -> NaN, zero_values -> 0, recode {old: new}, then multiply by scale.

    Args:
        X: numpy array (N, D) of raw values
        meta: dict from load_metadata
        names: list of the D column names of X

    Returns:
        numpy array (N, D)
    """
    X_clean = np.array(X, dtype=float)

    for j, name in enumerate(names):
        col_meta = meta[name]
        raw = X[:, j]
        col_data = X_clean[:, j]

        # Conversion na_values -> NaN
        col_data[np.isin(raw, col_meta["na_values"])] = np.nan

        # Conversion zero_values -> 0.0
        col_data[np.isin(raw, col_meta["zero_values"])] = 0.0

        # Recoding old values -> new values (keys are strings in the metadata)
        for old_val, new_val in col_meta["recode"].items():
            col_data[raw == float(old_val)] = new_val

        # Multiplication by scale factor (NaNs stay NaN)
        col_data *= col_meta["scale"]

    return X_clean


def fit_impute(X):
    """Compute the imputation statistics on the training data only.

    Returns:
        stats needed by apply_impute (e.g. one value per column)
    """

    # Compute medians ignoring nans
    stats = np.nanmedian(X, axis=0)

    # If a column is all nans, set its median to be 0.0
    stats = np.nan_to_num(stats, nan=0.0)

    return stats


def apply_impute(X, stats):
    """Fill the NaNs of X with the statistics computed by fit_impute.

    Returns:
        numpy array without NaNs
    """
    X_imputed = X.copy()

    # Replace nans with the statistics of the corresponding column
    X_imputed = np.where(np.isnan(X_imputed), stats, X_imputed)

    return X_imputed
