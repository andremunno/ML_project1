"""Check SGD updates, numerical stability, learning and evaluation isolation."""

import unittest

import numpy as np

from implementations import sigmoid
from run import stratified_split
from src.logistic import LogisticRegressionSGD


class LogisticTests(unittest.TestCase):
    def test_stable_extreme_probabilities(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_allclose(sigmoid([-10000, 0, 10000]), [0, 0.5, 1])

    def test_single_full_batch_matches_gradient(self):
        X = np.array([[1., 2.], [-2., 1.], [0., -1.]])
        y = np.array([1, -1, -1])
        residual = 0.5 - (y == 1)
        model = LogisticRegressionSGD(epochs=1, batch_size=10, learning_rate=0.2).fit(X, y)
        np.testing.assert_allclose(model.weights_, -0.2 * X.T @ residual / 3)
        self.assertAlmostEqual(model.intercept_, -0.2 * residual.mean())
        z = X @ model.weights_ + model.intercept_
        expected_loss = np.mean(np.logaddexp(0, z) - (y == 1) * z)
        self.assertAlmostEqual(model.loss_history_[0], expected_loss)

    def test_regularized_update_against_numerical_gradient(self):
        X = np.array([[1., 2.], [-2., 1.], [0., -1.]])
        y = np.array([1, 0, 0])
        first = LogisticRegressionSGD(epochs=1, batch_size=3, learning_rate=0.2,
                                      lambda_=0.3).fit(X, y)
        theta = np.r_[first.weights_, first.intercept_]

        def objective(t):
            z = X @ t[:-1] + t[-1]
            return np.mean(np.logaddexp(0, z) - y * z) + 0.3 * np.sum(t[:-1] ** 2) / 2

        gradient = np.zeros(3)
        for j in range(3):
            step = np.eye(3)[j] * 1e-6
            gradient[j] = (objective(theta + step) - objective(theta - step)) / 2e-6
        second = LogisticRegressionSGD(epochs=2, batch_size=3, learning_rate=0.2,
                                       lambda_=0.3).fit(X, y)
        np.testing.assert_allclose(np.r_[second.weights_, second.intercept_],
                                   theta - 0.2 / np.sqrt(2) * gradient, atol=1e-9)

    def test_learning_label_mapping_and_reproducibility(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(301, 3))  # deliberately not divisible by batch size
        y = np.where(X[:, 0] - X[:, 1] > 0.4, 1, -1)
        model = LogisticRegressionSGD(epochs=40, batch_size=32, learning_rate=0.2).fit(X, y)
        other = LogisticRegressionSGD(epochs=40, batch_size=32, learning_rate=0.2).fit(X, y == 1)
        np.testing.assert_array_equal(model.weights_, other.weights_)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])
        self.assertGreater(np.mean(model.predict(X) == y), 0.95)
        self.assertEqual(set(np.unique(model.predict(X))), {-1, 1})

    def test_constant_schedule_second_update(self):
        X = np.array([[1.], [-2.], [0.]])
        y = np.array([1, 0, 0])
        first = LogisticRegressionSGD(epochs=1, batch_size=3, learning_rate=0.2).fit(X, y)
        residual = first.predict_proba(X) - y
        second = LogisticRegressionSGD(epochs=2, batch_size=3, learning_rate=0.2,
                                       schedule="constant").fit(X, y)
        np.testing.assert_allclose(second.weights_, first.weights_ - 0.2 * X.T @ residual / 3)
        self.assertAlmostEqual(second.intercept_, first.intercept_ - 0.2 * residual.mean())
        with self.assertRaises(ValueError):
            LogisticRegressionSGD(schedule="unknown")

    def test_invalid_data(self):
        model = LogisticRegressionSGD()
        for y in ([1, 1], [-1, 0], [1, 2], [[1], [-1]]):
            with self.assertRaises(ValueError):
                model.fit(np.ones((2, 1)), y)
        with self.assertRaises(ValueError):
            model.fit(np.array([[np.nan], [1]]), [1, -1])
        with self.assertRaises(ValueError):
            model.predict(np.ones((2, 1)))

    def test_splits_are_disjoint_complete_and_stratified(self):
        y = np.r_[np.ones(100), -np.ones(900)]
        splits = stratified_split(y, 38)
        self.assertEqual([len(part) for part in splits], [700, 150, 150])
        np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(len(y)))
        for part, repeated in zip(splits, stratified_split(y, 38)):
            self.assertEqual(np.mean(y[part] == 1), 0.1)
            np.testing.assert_array_equal(part, repeated)


if __name__ == "__main__":
    unittest.main()
