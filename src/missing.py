"""Missing-value handling (Lorenzo)."""


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
    raise NotImplementedError


def fit_impute(X):
    """Compute the imputation statistics on the training data only.

    Returns:
        stats needed by apply_impute (e.g. one value per column)
    """
    raise NotImplementedError


def apply_impute(X, stats):
    """Fill the NaNs of X with the statistics computed by fit_impute.

    Returns:
        numpy array without NaNs
    """
    raise NotImplementedError
