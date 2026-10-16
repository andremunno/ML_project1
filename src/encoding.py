"""Encoding of categorical features (Andreas)."""


def fit_encode(X, meta, names):
    """Learn the encoding on the training data only (e.g. the one-hot categories).

    Binary -> 0/1, ordinal -> kept as number, nominal -> one-hot.

    Args:
        X: numpy array (N, D) without NaNs
        meta: dict from load_metadata
        names: list of the D column names of X

    Returns:
        encoder needed by apply_encode
    """
    raise NotImplementedError


def apply_encode(X, encoder):
    """Encode X with the encoder learned by fit_encode.

    Returns:
        X_encoded: numpy array (N, D')
        new_names: list of the D' column names
    """
    raise NotImplementedError
