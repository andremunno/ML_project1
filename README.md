# EPFL Machine Learning Project 1

## Team
- Andrea Munno
- Lorenzo Redighieri
- Andreas Caeyman

## Encode the supplied dataset

The executable preprocessing pipeline uses only NumPy and the Python standard
library. It writes numeric training/test matrices, matching feature names,
original labels and IDs, a reusable fitted encoder, and a per-variable audit.

```powershell
python encode_dataset.py --zip "C:\path\to\dataset_to_release.zip" --optimize
```

After the first extraction, reproduce the search/export with:

```powershell
python encode_dataset.py --optimize
python -m unittest discover -s tests -v
```

The default `extended` profile processes every available non-administrative
predictor marked `include` or `alternative` in the existing metadata. `--profile
core` restricts predictors to the team's existing `include` policy. Alternatives
are retained in the extended export to cover all substantive features; many
duplicate information, so ridge regularization or later feature selection is
appropriate. Administrative fields, IDs, survey weights, and target-leaking
questions are intentionally excluded and recorded in `feature_audit.csv`.

### Encoding decisions

| Feature meaning | Representation | Reason |
| --- | --- | --- |
| Binary responses | One 0/1 column; BRFSS yes=1, no=0 when codes are 1/2 | Avoids redundant dummy columns. Other two-level variables use the larger code as 1; the mapping is saved. |
| Nominal categories, e.g. state, marital status, employment, race, activity type | One-hot with the most frequent retained category omitted as reference | Numeric codes do not imply distances or an order. No target encoding or ordinal numbers for nominal variables. |
| Ordered levels, e.g. general health, education, income, pain | Compare flexible one-hot against a compact standardized numeric score | One-hot does not assume equal distances or a linear response. Numeric scores use the metadata's corrected codes. Selection is empirical. |
| Counts, e.g. children, unhealthy days, visits | Keep the numerical count, then median-impute and standardize | A count has meaningful distances. One-hot expansion would discard that structure and create many parameters. |
| Unit-prefixed frequencies | Decode into daily frequency; exercise/strength into weekly frequency | 107 and 207 encode units as well as magnitude. They cannot be compared as raw numbers. |
| HHMM activity durations; mixed-unit weight/height | Minutes; kilograms/meters | Remove numeric storage formats before modeling. |
| MMYYYY test/vaccination dates | Chronological month index | Preserves elapsed-month differences. Technical interview dates remain excluded. |
| Missing responses | Separate categorical level; numeric training median plus missing flag | Missing, zero, and a reference category remain distinct. A blank can mean a skipped module; it is not automatically 'no'. |
| Rare/unseen categories | Separate OTHER channel; compare minimum counts 1 and 50 | Pool low-support categories without assigning an artificial order. Binary levels are never pooled. |
| All-missing or constant training fields | Drop | No training signal. A field dropped at fit stays dropped on test data. |

Nonresponse and zero codes are **field-specific**, using
`data/metadata/variables_BRFSS_2015.csv`; there is no global rule that 7, 9, 77,
or 99 is missing. In particular, unhealthy-days 88 becomes zero, but 77/99
become missing. `ASYMPTOM=8` ('not at any time') becomes zero. `LONGWTCH=555`
gets a lifetime flag instead of being treated as 555 days. Asthma diagnosis at
age 10 or younger retains the metadata's proxy value 10 and gets an interval
flag; it is not an exact reported age. Fruit frequency 300 uses the CDC's
0.02/day convention. Capped values remain capped; missingness and interval
flags do not recover unobserved exact values.

`LASTSMK2=8` ('never smoked regularly') and vision difficulty code 6 ('unable
for other reasons') get separate flags; they are not assigned the highest
ordinal score. The original metadata's not-applicable policy is otherwise
preserved, with those values represented through the missingness channels.

Stored measurement scales already corrected in the supplied dataset are not
divided again. The metadata's specific scale corrections are still applied.
Use `src.brfss.prepare_raw` exactly once on raw values. `fit_encode` and
`apply_encode` expect cleaned values; do not run generic category imputation
first. The existing `src/missing.py` placeholders are separate team work and
are not used by this executable pipeline.

### Optimization and leakage control

`--optimize` compares four encodings (ordinal score/one-hot crossed with rare
count 1/50), each with ridge strengths 0.001, 0.01, and 0.1. A deterministic
stratified split uses approximately 30,000 fitting rows and 10,000 validation
rows, seed 38. Every encoder, median, scaling statistic, rare-level decision,
and reference category is fitted on the fitting portion only. A NumPy ridge
classifier serves as an inexpensive representation probe. Selection maximizes
validation ROC AUC. F1 and its threshold are also reported, but that threshold
is tuned on the same validation rows.

These are **tuning results**, not an unbiased estimate of final performance or
a claim of globally optimal encoding. They should be rechecked using the final
classifier and independent/nested validation. The test set does not influence
selection or fitted statistics. After selection, the encoder is refitted on all
training rows, so its output width may differ from the tuning sample. Refitting
the absolute frequency cutoff on a larger training set can retain more levels.
For cross-validation, call the fitting functions separately inside each fold;
do not split the already exported, full-training-fitted matrix to score models.

### Results on the supplied dataset

The completed extended export has **328,135 training rows, 109,379 test rows,
and 1,005 output columns**. The audit accounts for all 321 raw features:
277 substantive source fields are encoded (195 categorical, 38 discrete,
42 continuous and 2 dates); 44 fields are excluded under the existing policy.
These exclusions comprise 43 technical fields and `HAREHAB1`, whose question
availability can reveal a prior heart attack. The categorical count includes
both nominal and ordinal source fields.

For the strongest ridge setting in the tested grid (lambda 0.1):

| Ordinal representation | Rare minimum | Tuning dimensions | Validation AUC | Tuned F1 |
| --- | ---: | ---: | ---: | ---: |
| Numeric score | 1 | 1,038 | 0.845773 | 0.412560 |
| Numeric score | 50 | 904 | 0.845776 | 0.412888 |
| One-hot | 1 | 1,227 | 0.845196 | 0.408753 |
| One-hot | 50 | 1,045 | 0.845196 | 0.408777 |

The selected score/minimum-50 setting is compact. Its advantage over the other
settings is small; the pooling variants are practically tied, and this single
split does not establish a statistically meaningful performance improvement.
The actual stratified split has 29,999 fitting and 9,999 validation rows because
class allocations are rounded down. Detailed results include all 12 trials.

### Outputs and use

All generated data remain local and ignored by Git under `data/encoded/`:

- `x_train.npy`, `x_test.npy`: finite float32 matrices with identical column order.
- `feature_names.json`: exact names in matrix-column order; no intercept included.
- `encoder.json`: fitted levels, reference categories, medians and scaling state.
- `raw_metadata.json`, `raw_feature_names.json`: input schema/cleaning metadata snapshot.
- `feature_audit.csv`: all original features, handling, width and exclusion rationale.
- `train_ids.npy`, `test_ids.npy`, `y_train.npy`: original row alignment; labels stay -1/+1.
- `validation_results.json`, `tuning_split.npz`: search results and reproducible split.
- `summary.json`: export dimensions and selected settings.

```python
import json
import numpy as np

X = np.load("data/encoded/x_train.npy", mmap_mode="r")
X_test = np.load("data/encoded/x_test.npy", mmap_mode="r")
y = np.load("data/encoded/y_train.npy")
names = json.load(open("data/encoded/feature_names.json"))
y01 = (y == 1).astype(float)  # For logistic regression only.
```

The CSV loader caches float64 raw arrays in `data/raw/` and exports to float32
memory-mapped files in batches. This avoids repeated CSV parsing and a huge
dense CSV of dummy variables. Full-data cleaning still needs a few GB of RAM.
The reserved missing/OTHER channels can be constant on training data: use ridge
or remove constant output columns within each training fold before unregularized
normal equations. Standardization affects numeric columns, not dummy columns.

The project description supplied by the user was consulted as context, including
its NumPy-only dependency constraint and its link to the official
[CDC 2015 BRFSS documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html).
Meanings and unit conversions were checked against the
[2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
and the [calculated-variable definitions](https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_Calculated_Variables_Version4_08_10_17-508c.pdf).
