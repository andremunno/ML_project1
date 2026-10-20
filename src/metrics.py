"""Evaluation metrics (Lorenzo). Labels in {-1, 1} or {0, 1}: 1 is the positive class."""

import numpy as np


def f1_score(y_true, y_pred):
    """F1 of the positive class (1)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true != 1))
    fn = np.sum((y_pred != 1) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def best_threshold(y_true, scores):
    """Threshold on the scores that maximizes F1 (predict 1 when score >= threshold).

    Returns:
        (best_f1, threshold)
    """
    positive = (y_true == 1).astype(float)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tp = np.cumsum(positive[order])
    k = np.arange(1, len(scores) + 1)
    f1 = 2 * tp / (k + positive.sum())
    # Only cut between different scores
    f1[np.r_[sorted_scores[1:] == sorted_scores[:-1], False]] = -1
    i = int(np.argmax(f1))
    return float(f1[i]), float(sorted_scores[i])


def auc(y_true, scores):
    """Area under the ROC curve (rank formula, average ranks for ties)."""
    positive = y_true == 1
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(scores))
    ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
    p, n = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - p * (p + 1) / 2) / (p * n))
