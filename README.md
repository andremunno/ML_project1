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
| `src/logistic.py` | Basic NumPy logistic classifier with mini-batch SGD |
| `data/metadata/variables_BRFSS_2015.csv` | Per-variable type, missing/zero codes, recodings, group and use |

## Dataset setup for teammates

The dataset is intentionally excluded from Git. Obtain `dataset_to_release.zip`
separately and extract the three CSV files from its `dataset/` directory into
`data/raw/` inside your local checkout. Create that directory if necessary.
The files must be directly inside `data/raw/`, not `data/raw/dataset/`:

```text
ML_project1/
  data/
    raw/
      x_train.csv
      y_train.csv
      x_test.csv
    metadata/
      variables_BRFSS_2015.csv   (already included in Git)
```

From the repository root, install NumPy and run:

```sh
python -m pip install numpy
python run.py
```

Python 3 is required. The first run creates `data/raw/cache.npz` automatically;
you do not need to download or share this cache or the `data/encoded/` files.
Predictions are written to `data/submissions/submission.csv` and experiment
results to `experiments/results/`. Run `python run.py` before
`python benchmark_logistic.py`, since the benchmark uses the generated cache.
If you replace the source CSVs, delete the old `data/raw/cache.npz` so it is rebuilt.

To store the CSVs elsewhere, pass their containing directory using
`python run.py --data-dir "/path/to/dataset"`. The benchmark uses the default
`data/raw/` location.

Git ignores local data directories other than the shared metadata, as well as
ZIP archives, NumPy arrays (`.npy`) and caches (`.npz`) wherever they are saved.
Share the dataset separately rather than committing it. Keep the small metadata
CSV in the repository because preprocessing requires it.

## Preprocessing

```
select_features -> clean -> decode -> encode -> standardize
```

1. **select_features**: variables with `use == include` and `group <= max_group` (groups 1 to 5).
2. **clean**: per variable, from the metadata: `na_values` to NaN, `zero_values` to 0, `recode`, `scale`.
3. **decode**: prefix-coded frequencies and durations to one unit, MMYYYY dates to a month index,
   special answers (e.g. `LASTSMK2 = 8`, never smoked regularly) to a separate 0/1 flag.
4. **encode**: binary variables to one 0/1 column; nominal variables to one-hot without the most
   frequent level, rare levels (< 50 in training) pooled in `__other`; ordinal and numeric variables
   median-imputed; a `__missing` flag for every column with missing values.
5. **standardize**: every encoded column (0/1 columns included) to mean 0 and standard deviation 1.

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

## Basic logistic regression

Requires Python 3 and NumPy (`python -m pip install numpy`). From the repository root:

```sh
python run.py
python run.py --evaluate-only
```

The default uses feature groups 1 and 2, 30 epochs, batches of 256, an initial
learning rate of 0.05, and seed 38. These are baseline settings, not tuned optima.
`--max-group`, `--epochs`, `--batch-size`, `--learning-rate`, `--schedule`, `--lambda`,
`--seed`, `--data-dir`, and `--output` can be changed. For example,
`python run.py --lambda 0.0001` adds mild L2 regularization. Training uses NumPy
only; no scikit-learn or SciPy optimizer is needed.

The script makes a stratified 70% / 15% / 15% split. Preprocessing is fitted on
the first part only; the second part selects a probability threshold maximizing
positive-class F1; the third part measures F1, accuracy and ROC AUC at both 0.5
and the selected threshold. An always-negative baseline is also reported.
Evaluation is saved to `experiments/results/logistic_baseline.json`.
The default command then refits preprocessing and the classifier on all labeled
rows and writes `data/submissions/submission.csv` with `Id,Prediction` and labels
in {-1, 1}. The selected threshold is transferred to this final fit; its exact
optimality can change after refitting. Reported holdout metrics describe the
split-trained model, not performance on the unlabeled test set.

The existing `data/encoded` files are not used for evaluation: preprocessing
must be learned inside the training split. The initial raw CSV load may take
some time; the existing loader writes `data/raw/cache.npz` for later runs.

### Model and optimizer choice

For an encoded, standardized row x, the model predicts
`p = sigmoid(x @ w + b)`. Labels are mapped to 0/1 internally. It minimizes:

```text
J(w, b) = mean(logaddexp(0, X @ w + b) - y01 * (X @ w + b))
          + lambda / 2 * ||w||^2
```

The intercept b is not penalized. The default lambda is zero for an ordinary
logistic regression baseline. Stable sigmoid and `logaddexp` calculations avoid
overflow for extreme logits.

**Mini-batch stochastic gradient descent** is a reasonable refinement of
single-example SGD for this dataset (328,135 labeled rows). For each shuffled
batch B, calculate the old-parameter residual `r = sigmoid(X_B @ w + b) - y_B`
and update both parameters:

```text
w <- w - eta * (X_B.T @ r / |B| + lambda * w)
b <- b - eta * mean(r)
eta = initial_learning_rate / sqrt(epoch + 1)
```

- Compared with one-example SGD, averaging 256 examples generally reduces
  gradient noise and makes better use of NumPy's matrix operations.
- Compared with full-batch GD, it updates parameters before scanning all rows.
  Each epoch still costs O(ND), but each update uses only O(batch_size * D)
  temporary storage. The current preprocessing still loads the dataset in memory.
- The learning rate decreases between epochs to reduce late-training oscillation.
  This finite 30-epoch schedule does not establish convergence to the exact minimum.
- Logistic loss is convex, so escaping local minima is not the motivation for SGD.
  Newton's method uses curvature but requires constructing and solving a Hessian
  system. L-BFGS is a strong alternative for precise optimization of a moderate
  feature set, but implementing line search and curvature updates adds complexity
  to a basic NumPy baseline. No optimizer speed comparison has been run here.

The per-example and full-batch tradeoff is also discussed in
[Stanford CS229 lecture notes](https://cs229.stanford.edu/summer2023/cs229-notes1.pdf).
F1 is used to select the decision threshold, not as the differentiable training
objective. The classifier starts with the natural class distribution and no
class reweighting; inspect F1 and AUC as well as accuracy because the labels are
imbalanced.

The six course functions and their signatures in `implementations.py` are
preserved; `logistic_regression` there remains a full-batch GD implementation.

### Baseline result

Run on 2026-10-24 with the defaults above: 328,135 labeled rows, 8.83% positive,
75 encoded features, and split sizes of 229,694 / 49,220 / 49,221.

| Decision rule | Held-out F1 | Held-out accuracy | Held-out ROC AUC |
| --- | ---: | ---: | ---: |
| Always negative | 0.0000 | 0.9117 | 0.5000 |
| Logistic, threshold 0.5 | 0.2208 | 0.9137 | 0.8536 |
| Logistic, tuning-selected threshold 0.2081 | 0.4188 | 0.8741 | 0.8536 |

The lower threshold improves positive-class F1 while decreasing overall
accuracy. These results come from one independent holdout, not cross-validation
or an optimizer benchmark. The training objective decreased from 0.227323 after
the first epoch to 0.223142 after epoch 30; small intervening increases are
normal for stochastic updates.

### Comparing batch sizes and learning-rate schedules

```sh
python benchmark_logistic.py
python run.py --schedule constant --evaluate-only
```

The benchmark compares batch sizes 32, 128, 256, 1024 and 4096, each with the
existing square-root decay and a constant learning rate. It uses 30 epochs,
initial learning rate 0.05, three shuffle seeds (38, 39, 40), the same fixed
training/tuning split, and a shared preprocessing state fitted on training rows.
Batch size and schedule are optimizer hyperparameters; the probabilistic model
and loss do not change. This small grid is a controlled comparison, not a search
for the best possible learning rate for each batch size.

Every run processes the same number of examples, but a larger batch means fewer
updates per epoch. Runtime includes fitting and the training-loss calculation
after every epoch; it excludes preprocessing and tuning metrics. Timings depend
on the machine and BLAS thread settings. JSON results retain individual losses
for every epoch, and CSV results contain means and standard deviations across
shuffle seeds in `experiments/results/logistic_batch_benchmark.*`.

The benchmark uses tuning log loss to compare optimization/generalization,
and also reports tuning AUC and F1. F1 thresholds are optimized on these same
tuning rows, so those F1 values are exploratory, not independent performance
estimates. The evaluation split is not used for this comparison, and the
existing submission is not overwritten. The previously reported baseline
evaluation remains a historical result for that configuration.

With `--schedule sqrt` (the default), the update rate is
`0.05 / sqrt(epoch + 1)`: 0.05 in epoch 1, 0.025 in epoch 4, 0.01 in epoch 25,
and approximately 0.00913 in epoch 30. A mini-batch gradient contains sampling
noise even near the optimum. Smaller late updates reduce parameter oscillation.
Decay is a heuristic choice, not a requirement of logistic regression: a small
constant rate can work, and overly fast decay can slow optimization before the
solution is good. Larger batches have less noise and can benefit less from decay
at a fixed epoch budget, since they already take fewer updates. These tradeoffs
are explained in the authors' chapters on
[mini-batches](https://github.com/d2l-ai/d2l-en/blob/master/chapter_optimization/minibatch-sgd.md)
and [learning-rate schedules](https://github.com/d2l-ai/d2l-en/blob/master/chapter_optimization/sgd.md).

Benchmark run on 2026-10-24 (30 fits; OpenBLAS and OMP thread counts set to 1).
The following are mean tuning log losses over the three shuffle seeds; lower
is better. Times cover the decaying-rate fits only.

| Batch size | Updates per epoch | Fit time (s) | Loss, square-root decay | Loss, constant rate |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 7,178 | 5.18 | 0.225025 | 0.227368 |
| 128 | 1,795 | 3.43 | 0.224750 | 0.225252 |
| 256 | 898 | 2.85 | 0.224744 | 0.224935 |
| 1,024 | 225 | 2.71 | 0.225128 | 0.224747 |
| 4,096 | 57 | 4.81 | 0.234079 | 0.225511 |

Batch 256 with decay remains a reasonable default. Batch 128 with decay and
batch 1,024 with a constant rate are essentially tied with it on tuning loss;
this single split does not establish a meaningful performance winner. Constant
rate with batch 1,024 took 2.65 seconds on average. Decay helps the smaller
batches at this initial rate, but hurts the larger ones at this epoch budget.
Batch 4,096 is also slower on this machine, showing that larger matrix
operations do not guarantee lower total runtime. These conclusions apply to
the tested rate of 0.05; jointly tuning rates and epoch budgets may change them.
The classifier and default submission settings have not been changed by this
comparison.
