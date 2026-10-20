"""BRFSS-specific decodings (Andreas). Input must already be cleaned by src.missing.clean."""

import numpy as np

# Code that is a different answer, not the highest level: separate 0/1 flag.
SPECIAL_BRANCHES = {
    "LASTSMK2": (8, "never_regular"),
    "VIDFCLT2": (6, "other_reason"),
    "VIREDIF3": (6, "other_reason"),
}


def _frequency(x, factors):
    """Prefix codes (e.g. 203 = 3 per week) to one unit; factors = {prefix: multiplier}."""
    result = np.full(x.shape, np.nan)
    result[x == 0] = 0
    for prefix, factor in factors.items():
        mask = (x > prefix * 100) & (x < prefix * 100 + 100)
        result[mask] = (x[mask] % 100) * factor
    return result


def _flag(x, condition):
    """0/1 flag that stays NaN where x is missing."""
    flag = condition.astype(float)
    flag[np.isnan(x)] = np.nan
    return flag


def decode(X, meta, names):
    """Decode the BRFSS storage formats that clean cannot handle.

    Returns:
        X_out, meta_out, names_out (flags are added as extra binary columns)
    """
    columns, meta_out, names_out = [], {}, []

    def add(name, x, row, type_label=None):
        columns.append(x)
        names_out.append(name)
        meta_out[name] = dict(row, type_label=type_label or row["type_label"])
        if type_label:
            meta_out[name]["ordinal"] = "0"

    for j, name in enumerate(names):
        row, x = meta[name], X[:, j].copy()

        if name in SPECIAL_BRANCHES:
            code, label = SPECIAL_BRANCHES[name]
            add(name + "__" + label, _flag(x, x == code), row, "categorical")
            x[x == code] = np.nan
        elif name in ("BLDSUGAR", "FEETCHK2"):
            # times per day / week / month / year -> times per day
            x = _frequency(x, {1: 1, 2: 1 / 7, 3: 1 / 30, 4: 1 / 365})
        elif name == "LONGWTCH":
            # 555 = all my life; durations in days / weeks / months / years -> days
            add(name + "__lifetime", _flag(x, x == 555), row, "categorical")
            x = _frequency(x, {1: 1, 2: 7, 3: 30, 4: 365})
        elif name == "ASTHMAGE":
            # clean recodes 97 (age 10 or younger) to 10
            add(name + "__age_le_10", _flag(x, x == 10), row, "categorical")
        elif row["type_label"] == "date":
            # MMYYYY -> months since year 0
            month, year = np.floor(x / 10000), x % 10000
            valid = (month >= 1) & (month <= 12) & (year >= 1900)
            x = np.where(valid, year * 12 + month - 1, np.nan)
            add(name + "__month_index", x, row, "continuous")
            continue

        add(name, x, row)

    return np.column_stack(columns), meta_out, names_out
