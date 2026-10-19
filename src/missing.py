"""Missing-value handling (Lorenzo)."""

import numpy as np


def clean(X, meta, names):
    """Apply the metadata recodings, column by column.

    na_values -> NaN, zero_values -> 0, recode {old: new}, then multiply by scale.

    Args:
        X: numpy array (N, D)
        meta: dict from load_metadata
        names: list of the D column names of X

    Returns:
        numpy array (N, D)
    """

    X_clean = X.copy()

    for j, name in enumerate(names):
        # Check if the feature has defined metadata rules
        if name in meta:
            col_meta = meta[name]
            col_data = X_clean[:, j]

            # Conversion na_values -> NaN
            na_vals = col_meta.get("na_values")
            if na_vals:
                col_data[np.isin(col_data, na_vals)] = np.nan

            # Conversion zero_values -> 0.0
            zero_vals = col_meta.get("zero_values")
            if zero_vals:
                col_data[np.isin(col_data, zero_vals)] = 0.0

            # Recoding old values -> new values
            recode_dict = col_meta.get("recode")
            if recode_dict:
                for old_val_str, new_val in recode_dict.items():
                    # If the key is already a number, use it directly
                    if isinstance(old_val_str, (int, float)):
                        old_val = old_val_str
                    elif isinstance(old_val_str, str):
                        try:
                            # Check for decimal point only if it's a string
                            old_val = float(old_val_str) if '.' in old_val_str else int(old_val_str)
                        except ValueError:
                            old_val = old_val_str
                    else:
                        old_val = old_val_str
                    
                    col_data[col_data == old_val] = new_val

            # Multiplication by scale factor (NumPy handles multiplication with NaNs automatically)
            scale_val = col_meta.get("scale")
            if scale_val is not None and scale_val != 1.0:
                col_data = col_data * scale_val

            X_clean[:, j] = col_data

    return X_clean


def fit_impute(X):
    """Compute the imputation statistics on the training data only.

    Returns:
        stats needed by apply_impute (e.g. one value per column)
    """

    # Compute medians ignoring nans
    stats = np.nanmedian(X, axis=0)

    # If a column is all nans, set its median to be 0.0
    stats = np.nat_to_num(stats, nan=0.0)

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