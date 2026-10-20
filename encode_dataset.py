"""Reproducible NumPy-only BRFSS encoding/export. Run with --help."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from src.brfss import prepare_raw, select_predictors, transformation_note
from src.config import DATA_DIR, METADATA_PATH, ROOT, SEED
from src.encoding import fit_encode, apply_encode
from src.preprocessing import load_metadata


def read_csv(path):
    """Read numeric CSV in bounded batches, with a reusable on-disk cache."""
    path = Path(path)
    cache = path.with_suffix(".npy")
    with path.open(encoding="utf-8", newline="") as stream:
        names = next(csv.reader(stream))
    if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        with path.open("rb") as stream:
            n = sum(1 for line in stream if line.strip()) - 1
        partial = cache.with_suffix(".partial.npy")
        array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float64, shape=(n, len(names)))
        with path.open(encoding="utf-8", newline="") as stream:
            next(stream)
            batch, start = [], 0
            for line in stream:
                if not line.strip():
                    continue
                batch.append(line)
                if len(batch) == 4096:
                    array[start:start + len(batch)] = np.genfromtxt(batch, delimiter=",").reshape(-1, len(names))
                    start += len(batch)
                    batch = []
            if batch:
                array[start:start + len(batch)] = np.genfromtxt(batch, delimiter=",").reshape(-1, len(names))
        array.flush()
        del array
        partial.replace(cache)
    return np.load(cache, mmap_mode="r"), names


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def auc(y, scores):
    """Rank AUC with average ranks for ties."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
    p, n = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def best_f1(y, scores):
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    f1 = 2 * tp / (k + y.sum())
    # Evaluate only at the end of each equal-score group.
    ends = np.r_[scores[order][1:] != scores[order][:-1], True]
    f1[~ends] = -1
    i = int(np.argmax(f1))
    return float(f1[i]), float(scores[order][i])


def optimize(raw, names, meta, y, profile, output, fit_size, val_size):
    """Small stratified holdout search. Test rows are never inspected here.

    This is a tuning score, not an independent generalization estimate.
    Ridge is a cheap fixed-family probe of representation quality, not the
    final classifier. Its lambda is penalized in the average squared loss.
    """
    rng = np.random.default_rng(SEED)
    fit, val = [], []
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        fraction = len(idx) / len(y)
        nf, nv = int(fit_size * fraction), int(val_size * fraction)
        if nf + nv > len(idx):
            raise ValueError("Requested optimization sample exceeds training data")
        fit.extend(idx[:nf])
        val.extend(idx[nf:nf + nv])
    fit, val = np.array(fit), np.array(val)
    np.savez(output / "tuning_split.npz", fit_indices=fit, validation_indices=val)
    xf, mf, nf = prepare_raw(np.asarray(raw[fit, 1:]), meta, names)
    xv, _, _ = prepare_raw(np.asarray(raw[val, 1:]), meta, names)
    indices = select_predictors(nf, mf, profile)
    nf = [nf[j] for j in indices]
    xf, xv = xf[:, indices], xv[:, indices]
    results = []
    for ordinal, minimum in [("score", 1), ("score", 50), ("onehot", 1), ("onehot", 50)]:
        state = fit_encode(xf, mf, nf, ordinal=ordinal, min_frequency=minimum)
        a = apply_encode(xf, state)[0].astype(np.float64)
        b = apply_encode(xv, state)[0].astype(np.float64)
        a = np.column_stack([np.ones(len(a)), a])
        b = np.column_stack([np.ones(len(b)), b])
        gram, rhs = a.T @ a / len(a), a.T @ y[fit] / len(a)
        for strength in (0.001, 0.01, 0.1):
            penalty = np.eye(gram.shape[0]) * strength
            penalty[0, 0] = 0
            w = np.linalg.solve(gram + penalty, rhs)
            scores = b @ w
            f1, threshold = best_f1(y[val], scores)
            result = dict(ordinal=ordinal, min_frequency=minimum, ridge_lambda=strength,
                          dimensions=a.shape[1] - 1, auc=auc(y[val], scores),
                          f1_tuned=f1, threshold_tuned=threshold)
            results.append(result)
            print(json.dumps(result), flush=True)
    # AUC avoids coupling the representation choice to a particular threshold.
    winner = max(results, key=lambda r: r["auc"])
    save_json(output / "validation_results.json", {
        "seed": SEED, "fit_rows": len(fit), "validation_rows": len(val),
        "profile": profile, "selection_metric": "auc", "winner": winner,
        "caveat": "Single holdout tuning scores, not independent test estimates. F1 thresholds also tuned here.",
        "results": results})
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Optional supplied dataset ZIP to extract")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "encoded")
    parser.add_argument("--profile", choices=["core", "extended"], default="extended")
    parser.add_argument("--ordinal", choices=["score", "onehot"], default="onehot")
    parser.add_argument("--min-frequency", type=int, default=50)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--fit-size", type=int, default=30000)
    parser.add_argument("--val-size", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.zip:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as archive:
            for filename in ("x_train.csv", "x_test.csv", "y_train.csv"):
                target = args.data_dir / filename
                if not target.exists():
                    with archive.open("dataset/" + filename) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    train, train_names = read_csv(args.data_dir / "x_train.csv")
    labels, label_names = read_csv(args.data_dir / "y_train.csv")
    if not np.array_equal(train[:, 0], labels[:, 0]):
        raise ValueError("Training IDs and label IDs are not aligned")
    y_original = np.asarray(labels[:, 1], dtype=np.int8)
    if not set(np.unique(y_original)) <= {-1, 1}:
        raise ValueError("Expected supplied -1/+1 labels")
    y = (y_original == 1).astype(float)
    names, meta = train_names[1:], load_metadata(args.metadata)
    if set(names) - meta.keys():
        raise ValueError("Missing metadata: " + str(set(names) - meta.keys()))
    choice = {"ordinal": args.ordinal, "min_frequency": args.min_frequency}
    if args.optimize:
        choice = optimize(train, names, meta, y, args.profile, args.output, args.fit_size, args.val_size)
    print("Fitting selected encoder on all training rows", flush=True)
    cleaned, expanded_meta, expanded_names = prepare_raw(train[:, 1:], meta, names)
    indices = select_predictors(expanded_names, expanded_meta, args.profile)
    selected_names = [expanded_names[j] for j in indices]
    selected = cleaned[:, indices]
    del cleaned
    encoder = fit_encode(selected, expanded_meta, selected_names,
                         ordinal=choice["ordinal"], min_frequency=choice["min_frequency"])
    save_json(args.output / "encoder.json", encoder)
    save_json(args.output / "feature_names.json", encoder["output_names"])
    save_json(args.output / "raw_metadata.json", meta)
    save_json(args.output / "raw_feature_names.json", names)
    # Export a complete per-input inventory, including intentional exclusions.
    audit = []
    specs = {s["name"]: s for s in encoder["features"]}
    for name in names:
        derivatives = [n for n in expanded_names if n == name or n.startswith(name + "__")]
        used = [specs[n] for n in derivatives if n in specs]
        row = meta[name]
        audit.append({"variable": name, "type": row["type_label"], "use": row["use"],
                      "status": "encoded" if any(s["kind"] != "drop" for s in used) else "excluded",
                      "encoding": "; ".join(s["name"] + ":" + s["kind"] for s in used),
                      "output_columns": sum(s["stop"] - s["start"] for s in used),
                      "raw_transformation": transformation_note(name),
                      "drop_reason": "; ".join(s.get("reason", "") for s in used if s["kind"] == "drop"),
                      "notes": row["notes"], "rationale": row["rationale"],
                      "na_values": json.dumps(row["na_values"]),
                      "zero_values": json.dumps(row["zero_values"]),
                      "recode": json.dumps(row["recode"])})
    with (args.output / "feature_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = {"profile": args.profile, "ordinal": choice["ordinal"],
               "min_frequency": choice["min_frequency"], "raw_features": len(names),
               "encoded_features": len(encoder["output_names"]), "dtype": "float32",
               "metadata_path": str(args.metadata.resolve()), "seed": SEED}
    for split in ("train", "test"):
        if split == "train":
            raw, headers = train, train_names
        else:
            del selected
            raw, headers = read_csv(args.data_dir / "x_test.csv")
            if headers != train_names:
                raise ValueError("Train/test column names or order differ")
            cleaned, _, test_names = prepare_raw(raw[:, 1:], meta, names)
            if test_names != expanded_names:
                raise ValueError("Expanded column mismatch")
            selected = cleaned[:, indices]
            del cleaned
        shape = (len(raw), len(encoder["output_names"]))
        dest = np.lib.format.open_memmap(args.output / ("x_" + split + ".npy"),
                                         mode="w+", dtype=np.float32, shape=shape)
        # Batching avoids a full-size finite check and temporary category masks.
        for start in range(0, len(raw), 8192):
            apply_encode(selected[start:start + 8192], encoder, out=dest[start:start + 8192])
        dest.flush()
        if not all(np.isfinite(dest[start:start + 8192]).all() for start in range(0, len(raw), 8192)):
            raise ValueError("Non-finite exported values")
        summary[split + "_shape"] = list(shape)
        np.save(args.output / (split + "_ids.npy"), np.asarray(raw[:, 0], dtype=np.int64))
        del dest
        print(f"Exported {split}: {shape}", flush=True)
    np.save(args.output / "y_train.npy", y_original)
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
