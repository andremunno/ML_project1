"""Cross-validation (Lorenzo)."""


def stratified_k_fold(y, k, seed):
    """Split the row indices into k folds with the same share of positives.

    Returns:
        list of k (train_idx, val_idx) pairs
    """
    raise NotImplementedError


def cross_validate(X_raw, y, names, meta, max_group, train_fn, k, seed):
    """k-fold CV of the full pipeline.

    For each fold:
      state = fit_preprocess(X_raw[tr], meta, names, max_group)   # train part only
      X_tr, _ = apply_preprocess(X_raw[tr], meta, names, state)
      X_va, _ = apply_preprocess(X_raw[va], meta, names, state)
      score = train_fn(y[tr], X_tr)          # returns a function X -> scores
      threshold from best_threshold(y[tr], score(X_tr))
      F1 on the validation part with that threshold

    Returns:
        dict with the per-fold and mean F1 (and the chosen thresholds)
    """
    raise NotImplementedError
