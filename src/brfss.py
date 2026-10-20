"""BRFSS-specific raw cleaning/unit decoding; NumPy only."""

import numpy as np

FRUIT = {"FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"}
WEEKLY = {"EXEROFT1", "EXEROFT2", "STRENGTH"}
SPECIAL_BRANCHES = {
    "LASTSMK2": (8, "never_regular"),
    "VIDFCLT2": (6, "other_reason"),
    "VIREDIF3": (6, "other_reason"),
}


def _frequency(x, factors):
    result = np.full(x.shape, np.nan)
    result[x == 0] = 0
    for prefix, factor in factors.items():
        mask = (x >= prefix * 100 + 1) & (x <= prefix * 100 + 99) & (x == np.floor(x))
        result[mask] = (x[mask] % 100) * factor
    return result


def prepare_raw(X, meta, names):
    """Clean RAW values ONCE and return (expanded X, metadata, names).

    Uses field-specific missing/zero/recode/scale metadata, not a global 7/9
    rule. Derived columns inherit their source's feature-selection policy.
    """
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("Raw width must match names")
    columns, result_meta, result_names = [], {}, []

    def add(name, x, row, kind=None):
        result_names.append(name)
        columns.append(x)
        result_meta[name] = dict(row)
        if kind:
            result_meta[name]["encoding_kind"] = kind

    for j, name in enumerate(names):
        row, raw = meta[name], X[:, j]
        x = raw.astype(np.float64, copy=True)
        x[~np.isfinite(x) | np.isin(raw, row["na_values"])] = np.nan
        x[np.isin(raw, row["zero_values"])] = 0
        for old, new in row["recode"].items():
            x[(raw == float(old)) & np.isfinite(x)] = float(new)
        x *= row["scale"]
        if name in SPECIAL_BRANCHES:
            code, label = SPECIAL_BRANCHES[name]
            flag = (raw == code).astype(float)
            flag[~np.isfinite(x)] = np.nan
            add(name + "__" + label, flag, row, "discrete")
            # A different reason is not the largest ordinal severity/interval.
            x[raw == code] = np.nan
        if name == "ASYMPTOM":
            x[x == 8] = 0  # 'Not at any time' precedes increasing symptom levels.
        if name == "ALCDAY5":
            x = _frequency(x, {1: 1 / 7, 2: 1 / 30})
        elif name in FRUIT:
            x = _frequency(x, {1: 1, 2: 1 / 7, 3: 1 / 30})
            x[raw == 300] = 0.02  # CDC calculated-frequency convention.
        elif name in WEEKLY:
            x = _frequency(x, {1: 1, 2: 7 / 30})
        elif name in {"BLDSUGAR", "FEETCHK2"}:
            x = _frequency(x, {1: 1, 2: 1 / 7, 3: 1 / 30, 4: 1 / 365})
        elif name in {"EXERHMM1", "EXERHMM2"}:
            valid = (x >= 1) & (x <= 959) & (x % 100 < 60) & (x == np.floor(x))
            x = np.where(valid, np.floor(x / 100) * 60 + x % 100, np.nan)
        elif name == "LONGWTCH":
            life = (raw == 555).astype(float)
            life[~np.isfinite(x)] = np.nan
            add(name + "__lifetime", life, row, "discrete")
            x = _frequency(x, {1: 1, 2: 7, 3: 30, 4: 365})
        elif name == "ASTHMAGE":
            young = (raw == 97).astype(float)
            young[np.isnan(x)] = np.nan
            add(name + "__age_le_10", young, row, "discrete")
        elif name in {"WEIGHT2", "HEIGHT3"}:
            if name == "WEIGHT2":
                x = np.where(x >= 9000, x - 9000, x * 0.45359237)
            else:
                x = np.where(x >= 9000, (x - 9000) / 100,
                             (np.floor(x / 100) * 12 + x % 100) * 0.0254)
        elif row["type_label"] == "date":
            if name == "IDATE":
                add(name, x, row)  # Technical field excluded by selection.
                continue
            month, year = np.floor(x / 10000), x % 10000
            valid = (month >= 1) & (month <= 12) & (year >= 1900) & (year <= 2016)
            add(name + "__month_index", np.where(valid, year * 12 + month - 1, np.nan), row, "continuous")
            continue
        add(name, x, row)
    return np.column_stack(columns), result_meta, result_names


def transformation_note(name):
    """Human-readable supplement to the original metadata in the audit."""
    if name in SPECIAL_BRANCHES:
        code, label = SPECIAL_BRANCHES[name]
        return f"Code {code}: separate {label} flag; ordinal component becomes missing"
    if name == "ASYMPTOM":
        return "Reorder code 8 (no symptoms) to 0"
    if name in FRUIT:
        return "Prefix units to times/day; 300=0.02/day; zero codes to 0"
    if name in WEEKLY:
        return "Prefix units to times/week"
    if name in {"ALCDAY5", "BLDSUGAR", "FEETCHK2"}:
        return "Prefix units to times/day; zero codes to 0"
    if name in {"EXERHMM1", "EXERHMM2"}:
        return "HHMM to minutes; invalid minute components to missing"
    if name == "LONGWTCH":
        return "Prefix duration to days; 555 to lifetime flag with missing duration"
    if name == "ASTHMAGE":
        return "97 to proxy age 10 plus age_le_10 interval flag"
    if name in {"FLSHTMY2", "HIVTSTD3"}:
        return "MMYYYY to chronological month index"
    if name in {"HEIGHT3", "WEIGHT2"}:
        return "Mixed units to meters/kilograms"
    return "Metadata missing/zero/recode/scale rules"


def select_predictors(names, meta, profile="core"):
    """Core: existing include policy. Extended: include + alternative.

    Both exclude survey administration/weights, identifiers, text and leakage.
    Every raw column is recorded in the export's feature audit.
    """
    if profile not in {"core", "extended"}:
        raise ValueError("profile must be core or extended")
    uses = {"include"} if profile == "core" else {"include", "alternative"}
    return [j for j, n in enumerate(names) if meta[n]["use"] in uses]
