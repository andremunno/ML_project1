"""Encoding of the cleaned features (Andreas). Fit on training rows only.

- binary categorical  -> one 0/1 column (BRFSS 1 = yes, 2 = no)
- nominal categorical -> one-hot without the most frequent level (reference);
                         levels seen fewer than min_frequency times go to __other
- ordinal, discrete, continuous -> median imputation + standardization
- every column with NaNs in training also gets a __missing 0/1 column
"""

import numpy as np

from src.missing import fit_impute, apply_impute


def fit_encode(X, meta, names, min_frequency=50):
    """Learn the encoding of each column on the training data.

    Args:
        X: numpy array (N, D), cleaned (NaN = missing)
        meta: dict from load_metadata (or from decode)
        names: list of the D column names of X

    Returns:
        encoder: dict used by apply_encode
    """
    features, out_names = [], []
    for j, name in enumerate(names):
        x = X[:, j]
        values, counts = np.unique(x[~np.isnan(x)], return_counts=True)
        has_missing = bool(np.isnan(x).any())
        if len(values) == 0 or (len(values) == 1 and not has_missing):
            continue  # no information in training

        spec = {"name": name, "col": j, "start": len(out_names), "missing": has_missing}
        nominal = meta[name]["type_label"] == "categorical" and meta[name]["ordinal"] != "1"

        if nominal and len(values) == 2:
            spec["kind"] = "binary"
            spec["positive"] = 1.0 if values.tolist() == [1.0, 2.0] else float(values[-1])
            out_names.append(name)
        elif nominal:
            keep = values[counts >= min_frequency]
            if len(keep) == 0:
                keep = values[[np.argmax(counts)]]
            baseline = keep[np.argmax(counts[np.isin(values, keep)])]
            spec["kind"] = "onehot"
            spec["known"] = [float(v) for v in keep]
            spec["levels"] = [v for v in spec["known"] if v != baseline]
            spec["other"] = len(keep) < len(values)  # some rare levels were pooled
            out_names += [f"{name}__{v:g}" for v in spec["levels"]]
            if spec["other"]:
                out_names.append(name + "__other")
        else:
            spec["kind"] = "numeric"
            spec["median"] = fit_impute(x[:, None]).tolist()
            filled = apply_impute(x[:, None], spec["median"])
            spec["mean"] = float(filled.mean())
            spec["std"] = float(filled.std()) or 1.0
            out_names.append(name)

        if has_missing:
            out_names.append(name + "__missing")
        features.append(spec)
    return {"features": features, "names": out_names}


def apply_encode(X, encoder):
    """Encode X (cleaned, same columns as in fit_encode).

    Returns:
        X_encoded: float32 numpy array (N, D')
        names: list of the D' column names
    """
    out = np.zeros((len(X), len(encoder["names"])), dtype=np.float32)
    for spec in encoder["features"]:
        x = X[:, spec["col"]]
        missing = np.isnan(x)
        k = spec["start"]

        if spec["kind"] == "binary":
            out[:, k] = x == spec["positive"]
            k += 1
        elif spec["kind"] == "onehot":
            for level in spec["levels"]:
                out[:, k] = x == level
                k += 1
            if spec["other"]:
                out[:, k] = ~missing & ~np.isin(x, spec["known"])
                k += 1
        else:
            filled = apply_impute(x[:, None], spec["median"])[:, 0]
            out[:, k] = (filled - spec["mean"]) / spec["std"]
            k += 1

        if spec["missing"]:
            out[:, k] = missing
    return out, list(encoder["names"])
