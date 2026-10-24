"""A NumPy logistic classifier trained with shuffled mini-batch SGD."""

import numpy as np

from implementations import sigmoid


class LogisticRegressionSGD:
    """Binary logistic regression; X must be imputed and standardized.

    Minimize mean binary cross-entropy + lambda_ * ||w||^2 / 2.
    The intercept is learned separately and is not regularized. Labels may be
    {-1, 1} or {0, 1}; predictions use the project's {-1, 1} convention.
    One epoch visits every training row once, including the final short batch.
    """

    def __init__(self, epochs=30, batch_size=256, learning_rate=0.05,
                 lambda_=0.0, seed=38, schedule="sqrt"):
        if epochs < 1 or batch_size < 1 or int(epochs) != epochs or int(batch_size) != batch_size:
            raise ValueError("epochs and batch_size must be positive integers")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError("lambda_ must be nonnegative and finite")
        if schedule not in ("sqrt", "constant"):
            raise ValueError("schedule must be 'sqrt' or 'constant'")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.seed = seed
        self.schedule = schedule

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("Expected nonempty X (N, D) and y (N,)")
        if not np.isfinite(X).all() or not np.isin(y, [-1, 0, 1]).all():
            raise ValueError("X must be finite and labels must be {-1, 1} or {0, 1}")
        if len(np.unique(y)) != 2 or (-1 in y and 0 in y):
            raise ValueError("Training requires both classes, encoded {-1, 1} or {0, 1}")
        target = (y == 1).astype(float)
        rng = np.random.default_rng(self.seed)
        self.weights_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.epochs):
            rate = self.learning_rate
            if self.schedule == "sqrt":
                rate /= np.sqrt(epoch + 1)
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                rows = order[start:start + self.batch_size]
                batch = X[rows]
                residual = sigmoid(batch @ self.weights_ + self.intercept_) - target[rows]
                gradient = batch.T @ residual / len(rows) + self.lambda_ * self.weights_
                self.weights_ -= rate * gradient
                self.intercept_ -= rate * residual.mean()

            # Chunked evaluation avoids a full float64 copy of X.
            total = 0.0
            for start in range(0, len(y), 8192):
                end = start + 8192
                z = X[start:end] @ self.weights_ + self.intercept_
                total += np.sum(np.logaddexp(0, z) - target[start:end] * z)
            loss = float(total / len(y) + self.lambda_ * (self.weights_ @ self.weights_) / 2)
            if not np.isfinite(loss):
                raise FloatingPointError("Training diverged; reduce learning_rate or check scaling")
            self.loss_history_.append(loss)
            if verbose:
                print(f"Epoch {epoch + 1:2d}/{self.epochs}: objective={loss:.6f}", flush=True)
        return self

    def predict_proba(self, X):
        """Return P(y=1 | X) as a one-dimensional array."""
        if not hasattr(self, "weights_"):
            raise ValueError("Fit the classifier before predicting")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.weights_) or not np.isfinite(X).all():
            raise ValueError("X must be finite with the same feature count as training")
        return sigmoid(X @ self.weights_ + self.intercept_)

    def predict(self, X, threshold=0.5):
        """Predict -1 or +1; threshold is a probability in [0, 1]."""
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return np.where(self.predict_proba(X) >= threshold, 1, -1)
