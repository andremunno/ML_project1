"""Compare batch sizes and learning-rate schedules on one fixed tuning split.

Uses three shuffle seeds, a shared training-fitted preprocessing state, and the
same number of epochs. Does not inspect evaluation labels or change submissions.
"""

import argparse
import csv
import json
from time import perf_counter

import numpy as np

from run import stratified_split
from src.config import DATA_DIR, RESULTS_DIR, SEED
from src.logistic import LogisticRegressionSGD
from src.metrics import auc, best_threshold
from src.preprocessing import apply_preprocess, fit_preprocess, load_feature_names, load_metadata


def log_loss(y, probabilities):
    p = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 256, 1024, 4096])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[38, 39, 40])
    args = parser.parse_args()
    # Load only labeled raw data; use the same cache as the baseline script.
    print("Loading labeled data and fitting shared preprocessing...", flush=True)
    cache = DATA_DIR / "cache.npz"
    if not cache.exists():
        parser.error("Run run.py once to populate data/raw/cache.npz")
    with np.load(cache) as data:
        raw, y = data["arr_0"], data["arr_2"]
    fit_rows, tune_rows, _ = stratified_split(y, SEED)
    names, meta = load_feature_names(), load_metadata()
    state = fit_preprocess(raw[fit_rows], meta, names, max_group=2)
    X_fit, feature_names = apply_preprocess(raw[fit_rows], meta, names, state)
    X_tune, _ = apply_preprocess(raw[tune_rows], meta, names, state)
    del raw
    runs, summaries = [], []
    for batch_size in args.batch_sizes:
        for schedule in ("sqrt", "constant"):
            group = []
            for seed in args.seeds:
                model = LogisticRegressionSGD(epochs=args.epochs, batch_size=batch_size,
                    learning_rate=args.learning_rate, seed=seed, schedule=schedule)
                started = perf_counter()
                model.fit(X_fit, y[fit_rows])
                seconds = perf_counter() - started
                p = model.predict_proba(X_tune)
                f1, threshold = best_threshold(y[tune_rows], p)
                run = {"batch_size": batch_size, "schedule": schedule, "seed": seed,
                       "seconds": seconds, "train_loss": model.loss_history_[-1],
                       "tuning_loss": log_loss(y[tune_rows], p),
                       "tuning_auc": auc(y[tune_rows], p), "tuning_f1": f1,
                       "threshold": threshold, "loss_history": model.loss_history_}
                runs.append(run)
                group.append(run)
                print(f"batch={batch_size:4d} {schedule:8s} seed={seed}: "
                      f"{seconds:.2f}s train={run['train_loss']:.6f} "
                      f"tuning={run['tuning_loss']:.6f}", flush=True)
            summary = {"batch_size": batch_size, "schedule": schedule,
                       "updates_per_epoch": int(np.ceil(len(fit_rows) / batch_size))}
            for metric in ("seconds", "train_loss", "tuning_loss", "tuning_auc", "tuning_f1"):
                summary[metric + "_mean"] = float(np.mean([r[metric] for r in group]))
                summary[metric + "_std"] = float(np.std([r[metric] for r in group]))
            summaries.append(summary)
            # Save completed groups so partial results survive an interrupted run.
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            result = {"epochs": args.epochs, "learning_rate": args.learning_rate,
                      "seeds": args.seeds, "split_seed": SEED, "max_group": 2,
                      "n_fit": len(fit_rows), "n_tune": len(tune_rows),
                      "features": len(feature_names),
                      "timing_scope": "fit only, including per-epoch training loss",
                      "note": "F1 threshold chosen and scored on tuning rows; exploratory only. "
                              "Evaluation split unused. Learning rate not tuned per batch size.",
                      "runs": runs, "summary": summaries}
            (RESULTS_DIR / "logistic_batch_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
            with (RESULTS_DIR / "logistic_batch_benchmark.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
    print("\nMean results (three seeds by default):", flush=True)
    for r in summaries:
        print(f"{r['batch_size']:4d} {r['schedule']:8s} "
              f"{r['seconds_mean']:.2f}s tuning loss={r['tuning_loss_mean']:.6f} "
              f"AUC={r['tuning_auc_mean']:.4f} F1={r['tuning_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
