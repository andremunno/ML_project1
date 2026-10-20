"""Training-only encoding using NumPy. Input values must already be cleaned.

Supports NaNs directly. Do not impute categories before calling this module:
their missingness is represented explicitly. State is JSON serializable.
"""

import numpy as np


def _validate(X, names):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X must be 2-D with one column per name")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique")
    if np.isinf(X).any():
        raise ValueError("Replace infinite values with NaN before encoding")
    return X


def fit_encode(X, meta, names, *, ordinal="onehot", min_frequency=50):
    """Fit on CLEANED training rows only; never on validation/test rows.

    Nominal: reference-coded one-hot, pooling rare/unseen levels into OTHER.
    Binary: one column (BRFSS 1/2 uses yes=1/no=0). Ordinal: flexible one-hot
    or compact numeric score. Counts/measurements: median imputation/z-scaling.
    Constant/all-missing training columns are dropped. Missing/OTHER channels
    are reserved for categories even if absent at fit. No intercept is added.
    min_frequency is an absolute count; binary levels are never pooled.
    """
    X = _validate(X, names)
    if len(X) == 0 or ordinal not in {"onehot", "score"}:
        raise ValueError("Need nonempty rows and ordinal=onehot/score")
    if not isinstance(min_frequency, int) or min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer")
    specs, out_names = [], []
    for j, name in enumerate(names):
        row = meta[name]
        kind = row.get("encoding_kind", row["type_label"])
        x = X[:, j]
        values, counts = np.unique(x[np.isfinite(x)], return_counts=True)
        missing = bool(np.isnan(x).any())
        spec = {"name": name, "source": j, "start": len(out_names)}
        if kind in {"identifier", "text", "date"}:
            raise ValueError(f"{name}: prepare dates or exclude identifiers/text first")
        if len(values) == 0 or (len(values) == 1 and not missing):
            spec.update(kind="drop", reason="all missing or constant on training rows")
        elif kind == "categorical" and not (ordinal == "score" and "Ordinal" in row.get("notes", "")):
            if len(values) == 2:
                positive = 1.0 if values.tolist() == [1.0, 2.0] else float(values[-1])
                spec.update(kind="binary", levels=values.tolist(), positive=positive)
                out_names.append(name + "__binary")
            else:
                keep = values[counts >= min_frequency]
                if len(keep) == 0:
                    keep = values[[np.argmax(counts)]]
                baseline = float(keep[np.argmax([counts[np.searchsorted(values, v)] for v in keep])])
                spec.update(kind="onehot", levels=keep.tolist(), baseline=baseline)
                out_names.extend(name + "__eq_" + format(v, "g") for v in keep if v != baseline)
            out_names.extend([name + "__missing", name + "__other"])
        else:
            median = float(np.median(x[np.isfinite(x)]))
            filled = np.where(np.isnan(x), median, x).astype(np.float64)
            std = float(filled.std())
            spec.update(kind="numeric", median=median, mean=float(filled.mean()),
                        std=std if std > 1e-12 else 1.0)
            out_names.extend([name, name + "__missing"])
        spec["stop"] = len(out_names)
        specs.append(spec)
    return {"version": 1, "input_names": list(names), "output_names": out_names,
            "ordinal": ordinal, "min_frequency": min_frequency, "features": specs}


def apply_encode(X, encoder, *, out=None):
    """Transform cleaned rows in input_names order. Returns float32 X and names.

    out can be a preallocated float32 memmap to bound peak memory use.
    """
    X = _validate(X, encoder["input_names"])
    shape = (len(X), len(encoder["output_names"]))
    if out is None:
        out = np.zeros(shape, dtype=np.float32)
    elif out.shape != shape or out.dtype != np.float32:
        raise ValueError("out must have expected shape and dtype float32")
    else:
        out[:] = 0
    for spec in encoder["features"]:
        kind = spec["kind"]
        if kind == "drop":
            continue
        x = X[:, spec["source"]]
        start, stop = spec["start"], spec["stop"]
        missing = np.isnan(x)
        if kind == "numeric":
            out[:, start] = (np.where(missing, spec["median"], x) - spec["mean"]) / spec["std"]
            out[:, start + 1] = missing
        else:
            if kind == "binary":
                out[:, start] = x == spec["positive"]
            else:
                for k, level in enumerate(v for v in spec["levels"] if v != spec["baseline"]):
                    out[:, start + k] = x == level
            out[:, stop - 2] = missing
            out[:, stop - 1] = ~missing & ~np.isin(x, spec["levels"])
    return out, list(encoder["output_names"])
