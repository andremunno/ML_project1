# EPFL Machine Learning Project 1

## Team
- Andrea Munno
- Lorenzo Redighieri
- Andreas Caeyman

## Structure

| File | Content |
| --- | --- |
| `implementations.py` | The six required methods |
| `run.py` | Produces the final submission |
| `helpers.py` | Course helpers (`load_csv_data`, `create_csv_submission`) |
| `src/config.py` | Paths and seed |
| `src/preprocessing.py` | Data loading, feature selection, full pipeline |
| `src/missing.py` | Metadata cleaning and median imputation |
| `src/brfss.py` | BRFSS storage formats (prefix codes, dates, special answers) |
| `src/encoding.py` | Binary / one-hot / numeric encoding |
| `src/metrics.py` | F1, accuracy, best threshold, AUC |
| `src/validation.py` | Cross-validation |
| `data/metadata/variables_BRFSS_2015.csv` | Per-variable type, missing/zero codes, recodings, group and use |

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/raw/`.

## Preprocessing

```
select_features -> clean -> decode -> encode
```

1. **select_features**: variables with `use == include` and `group <= max_group` (groups 1 to 5).
2. **clean**: per variable, from the metadata: `na_values` to NaN, `zero_values` to 0, `recode`, `scale`.
3. **decode**: prefix-coded frequencies and durations to one unit, MMYYYY dates to a month index,
   special answers (e.g. `LASTSMK2 = 8`, never smoked regularly) to a separate 0/1 flag.
4. **encode**: binary variables to one 0/1 column; nominal variables to one-hot without the most
   frequent level, rare levels (< 50 in training) pooled in `__other`; ordinal and numeric variables
   median-imputed and standardized; a `__missing` flag for every column with missing values.

Everything learned from data (levels, medians, means, standard deviations) is fitted on the
training rows only:

```python
from src.preprocessing import load_data, load_feature_names, load_metadata
from src.preprocessing import fit_preprocess, apply_preprocess

x_train, x_test, y_train, train_ids, test_ids = load_data()
names, meta = load_feature_names(), load_metadata()

state = fit_preprocess(x_train, meta, names, max_group=2)
X_train, feature_names = apply_preprocess(x_train, meta, names, state)
X_test, _ = apply_preprocess(x_test, meta, names, state)
```

In cross-validation, call `fit_preprocess` inside each fold on the training part only.

## Tests

```
python -m unittest discover -s tests
```
