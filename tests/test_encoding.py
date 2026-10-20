"""Behavioral checks of the preprocessing pipeline. Run: python -m unittest discover -s tests"""

import unittest

import numpy as np

from src.brfss import decode
from src.encoding import apply_encode, fit_encode
from src.metrics import auc, best_threshold, f1_score
from src.missing import clean
from src.preprocessing import apply_preprocess, fit_preprocess, load_metadata, select_features

META = load_metadata()


def prepare(raw, names):
    return decode(clean(np.array(raw, dtype=float), META, names), META, names)


class CleanTests(unittest.TestCase):
    def test_metadata_codes(self):
        names = ["PHYSHLTH", "ASTHMAGE", "_DRNKWEK"]
        x = clean(np.array([[88, 97, 420], [77, 30, 99900], [5, 98, 0]], dtype=float), META, names)
        np.testing.assert_allclose(x[0], [0, 10, 4.2])
        self.assertTrue(np.isnan(x[1, 0]) and np.isnan(x[1, 2]) and np.isnan(x[2, 1]))
        self.assertEqual(x[1, 1], 30)


class DecodeTests(unittest.TestCase):
    def test_prefix_codes_and_flags(self):
        names = ["BLDSUGAR", "LONGWTCH", "ASTHMAGE", "FLSHTMY2"]
        x, meta, out = prepare([[407, 555, 97, 122015], [888, 203, 30, 12015]], names)
        self.assertEqual(out, ["BLDSUGAR", "LONGWTCH__lifetime", "LONGWTCH",
                               "ASTHMAGE__age_le_10", "ASTHMAGE", "FLSHTMY2__month_index"])
        np.testing.assert_allclose(x[:, 0], [7 / 365, 0])
        np.testing.assert_array_equal(x[:, 1], [1, 0])
        self.assertTrue(np.isnan(x[0, 2]))
        self.assertEqual(x[1, 2], 21)
        np.testing.assert_array_equal(x[:, 3], [1, 0])
        self.assertEqual(x[0, 5] - x[1, 5], 11)
        self.assertEqual(meta["LONGWTCH__lifetime"]["type_label"], "categorical")

    def test_special_branch_is_not_the_highest_level(self):
        x, _, out = prepare([[8], [7], [99]], ["LASTSMK2"])
        self.assertEqual(out, ["LASTSMK2__never_regular", "LASTSMK2"])
        np.testing.assert_array_equal(x[:2, 0], [1, 0])
        self.assertTrue(np.isnan(x[0, 1]) and np.isnan(x[2]).all())
        self.assertEqual(x[1, 1], 7)


class EncodingTests(unittest.TestCase):
    def test_rare_unseen_and_missing(self):
        meta = {"nominal": {"type_label": "categorical", "ordinal": "0"}}
        train = np.array([[1], [1], [1], [2], [2], [3], [np.nan]])
        state = fit_encode(train, meta, ["nominal"], min_frequency=2)
        encoded, names = apply_encode(np.array([[1], [2], [3], [99], [np.nan]]), state)
        self.assertEqual(names, ["nominal__2", "nominal__other", "nominal__missing"])
        np.testing.assert_array_equal(encoded, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_binary_numeric_and_constant(self):
        meta = {"yes": {"type_label": "categorical", "ordinal": "0"},
                "count": {"type_label": "discrete", "ordinal": "0"},
                "fixed": {"type_label": "categorical", "ordinal": "0"}}
        train = np.array([[1, 0, 5], [2, 2, 5], [np.nan, np.nan, 5]])
        state = fit_encode(train, meta, ["yes", "count", "fixed"])
        x, names = apply_encode(np.array([[1, np.nan, 5], [2, 100, 5]]), state)
        self.assertEqual(names, ["yes", "yes__missing", "count", "count__missing"])
        np.testing.assert_array_equal(x[:, 0], [1, 0])
        self.assertEqual(x[0, 2], 0)  # median 1 equals the mean of the filled column
        np.testing.assert_array_equal(x[:, 3], [1, 0])


class PipelineTests(unittest.TestCase):
    def test_selection_and_pipeline(self):
        names = ["GENHLTH", "HAREHAB1", "SEQNO", "SEX", "_BMI5"]
        self.assertEqual(select_features(META, names, 1), [3, 4])
        self.assertEqual(select_features(META, names, 2), [0, 3, 4])
        raw = np.array([[1, 1, 7, 1, 2500], [3, 2, 8, 2, np.nan], [9, 1, 9, 1, 3000]] * 20)
        state = fit_preprocess(raw, META, names, max_group=2)
        x, out = apply_preprocess(raw[:3], META, names, state)
        self.assertEqual(out, ["GENHLTH", "GENHLTH__missing", "SEX", "_BMI5", "_BMI5__missing"])
        self.assertTrue(np.isfinite(x).all())


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        y = np.array([-1, 1, -1, 1])
        self.assertEqual(f1_score(y, np.array([-1, 1, 1, 1])), 0.8)
        self.assertEqual(auc(y, np.array([0, 1, 0, 1])), 1)
        self.assertEqual(auc(y, np.ones(4)), 0.5)
        self.assertEqual(best_threshold(y, np.array([0.1, 0.9, 0.2, 0.8])), (1.0, 0.8))


if __name__ == "__main__":
    unittest.main()
