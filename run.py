"""Train/evaluate a basic logistic classifier and write the project submission."""

import argparse
import json
from pathlib import Path

import numpy as np

from helpers import create_csv_submission
from src.config import DATA_DIR, RESULTS_DIR, SEED, SUBMISSION_PATH
from src.logistic import LogisticRegressionSGD
from src.metrics import accuracy, auc, best_threshold, f1_score
from src.preprocessing import (
    apply_preprocess, fit_preprocess, load_data, load_feature_names, load_metadata,
)


def stratified_split(y, seed):
    """70% fitting / 15% threshold tuning / 15% independent evaluation."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == label))
        a, b = int(0.70 * len(rows)), int(0.85 * len(rows))
        if min(a, b - a, len(rows) - b) == 0:
            raise ValueError("Each class needs enough rows for three nonempty splits")
        for part, indices in zip(parts, (rows[:a], rows[a:b], rows[b:])):
            part.extend(indices)
    return tuple(rng.permutation(part) for part in parts)


def evaluate(y, probabilities, threshold):
    predictions = np.where(probabilities >= threshold, 1, -1)
    return {"f1": float(f1_score(y, predictions)),
            "accuracy": float(accuracy(y, predictions)),
            "auc": float(auc(y, probabilities))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--schedule", choices=("sqrt", "constant"), default="sqrt")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--max-group", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    def new_model():
        return LogisticRegressionSGD(args.epochs, args.batch_size, args.learning_rate,
                                     args.lambda_, args.seed, args.schedule)

    print("Loading raw data (the first CSV load is cached)...", flush=True)
    raw, raw_test, y, _, test_ids = load_data(args.data_dir)
    names, meta = load_feature_names(args.data_dir), load_metadata()
    fit_rows, tune_rows, eval_rows = stratified_split(y, args.seed)
    print(f"Rows: {len(y)}; positive share: {np.mean(y == 1):.2%}", flush=True)
    print("Fitting preprocessing on the 70% training split...", flush=True)
    state = fit_preprocess(raw[fit_rows], meta, names, args.max_group)
    X_fit, feature_names = apply_preprocess(raw[fit_rows], meta, names, state)
    X_tune, _ = apply_preprocess(raw[tune_rows], meta, names, state)
    X_eval, _ = apply_preprocess(raw[eval_rows], meta, names, state)
    print(f"Encoded features: {len(feature_names)}", flush=True)
    model = new_model().fit(X_fit, y[fit_rows], verbose=True)
    _, threshold = best_threshold(y[tune_rows], model.predict_proba(X_tune))
    probabilities = model.predict_proba(X_eval)
    report = {
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "split_sizes": [len(fit_rows), len(tune_rows), len(eval_rows)],
        "positive_share": float(np.mean(y == 1)),
        "features": len(feature_names),
        "threshold": threshold,
        "evaluation_default_threshold": evaluate(y[eval_rows], probabilities, 0.5),
        "evaluation_tuned_threshold": evaluate(y[eval_rows], probabilities, threshold),
        "majority_baseline": evaluate(y[eval_rows], np.zeros(len(eval_rows)), 0.5),
        "training_objective": model.loss_history_,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "logistic_baseline.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "threshold", "evaluation_default_threshold", "evaluation_tuned_threshold",
        "majority_baseline")}, indent=2), flush=True)
    print(f"Evaluation saved to {report_path}", flush=True)
    if args.evaluate_only:
        return

    del X_fit, X_tune, X_eval
    print("Refitting preprocessing and classifier on all labeled rows...", flush=True)
    state = fit_preprocess(raw, meta, names, args.max_group)
    X, _ = apply_preprocess(raw, meta, names, state)
    final_model = new_model().fit(X, y, verbose=True)
    del X
    X_test, _ = apply_preprocess(raw_test, meta, names, state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, final_model.predict(X_test, threshold), args.output)
    print(f"Submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
