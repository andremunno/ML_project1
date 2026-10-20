"""Behavioral checks for category semantics, raw units, and frozen state."""

import json
import unittest

import numpy as np

from encode_dataset import auc, best_f1
from src.brfss import prepare_raw, select_predictors
from src.encoding import fit_encode, apply_encode
from src.preprocessing import load_metadata


class EncodingTests(unittest.TestCase):
    def test_rare_unseen_missing_and_frozen_state(self):
        meta = {"nominal": {"type_label": "categorical"}}
        train = np.array([[1], [1], [1], [2], [2], [3], [np.nan]])
        state = fit_encode(train, meta, ["nominal"], min_frequency=2)
        frozen = json.dumps(state)
        encoded, names = apply_encode(np.array([[1], [2], [3], [99], [np.nan]]), state)
        self.assertEqual(names, ["nominal__eq_2", "nominal__missing", "nominal__other"])
        np.testing.assert_array_equal(encoded, [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]])
        self.assertEqual(json.dumps(state), frozen)
        self.assertEqual(encoded.dtype, np.float32)

    def test_binary_and_numeric_training_statistics(self):
        meta = {"yes": {"type_label": "categorical"}, "count": {"type_label": "discrete"}}
        train = np.array([[1, 0], [2, 2], [np.nan, np.nan]])
        state = fit_encode(train, meta, ["yes", "count"])
        result, names = apply_encode(np.array([[1, np.nan], [2, 100], [8, 1]]), state)
        np.testing.assert_array_equal(result[:, names.index("yes__binary")], [1, 0, 0])
        np.testing.assert_array_equal(result[:, names.index("yes__other")], [0, 0, 1])
        self.assertEqual(result[0, names.index("count")], 0)
        self.assertEqual(result[0, names.index("count__missing")], 1)
        self.assertEqual(state["features"][1]["median"], 1)

    def test_empty_and_constant_fields(self):
        meta = {n: {"type_label": "categorical"} for n in ["empty", "fixed", "partial"]}
        x = np.array([[np.nan, 1, 1], [np.nan, 1, np.nan]])
        state = fit_encode(x, meta, list(meta))
        self.assertEqual([s["kind"] for s in state["features"]], ["drop", "drop", "onehot"])
        self.assertEqual(apply_encode(x, state)[0].shape, (2, 2))

    def test_units_and_special_codes(self):
        meta = load_metadata()
        names = ["PHYSHLTH", "FRUIT1", "ALCDAY5", "BLDSUGAR", "EXERHMM1", "ASYMPTOM", "FLSHTMY2"]
        raw = np.array([[88, 207, 107, 407, 130, 8, 122015],
                        [77, 555, 888, 888, 777, 1, 777777],
                        [5, 300, 230, 101, 45, 5, 12015]])
        x, _, expanded = prepare_raw(raw, meta, names)
        np.testing.assert_allclose(x[0, :6], [0, 1, 1, 7 / 365, 90, 0])
        self.assertTrue(np.isnan(x[1, 0]))
        np.testing.assert_allclose(x[1, 1:4], [0, 0, 0])
        self.assertTrue(np.isnan(x[1, 4]))
        self.assertEqual(x[2, 1], 0.02)
        self.assertEqual(x[0, -1] - x[2, -1], 11)
        self.assertIn("FLSHTMY2__month_index", expanded)
        np.testing.assert_array_equal(raw[:, 0], [88, 77, 5])

    def test_exclusion_and_metrics(self):
        meta = load_metadata()
        names = ["GENHLTH", "HAREHAB1", "SEQNO", "_AGEG5YR"]
        self.assertEqual(select_predictors(names, meta, "extended"), [0, 3])
        self.assertEqual(select_predictors(names, meta, "core"), [0])
        y = np.array([0, 1, 0, 1])
        self.assertEqual(auc(y, np.array([0, 1, 0, 1])), 1)
        self.assertEqual(auc(y, np.ones(4)), 0.5)
        self.assertEqual(best_f1(y, np.array([0, 1, 0, 1]))[0], 1)

    def test_nonordinal_branches_do_not_become_extreme_scores(self):
        meta = load_metadata()
        names = ["LASTSMK2", "VIREDIF3"]
        x, _, expanded = prepare_raw(np.array([[8, 6], [7, 5], [99, 7]]), meta, names)
        self.assertEqual(expanded, ["LASTSMK2__never_regular", "LASTSMK2",
                                    "VIREDIF3__other_reason", "VIREDIF3"])
        np.testing.assert_array_equal(x[0, [0, 2]], [1, 1])
        self.assertTrue(np.isnan(x[0, [1, 3]]).all())
        np.testing.assert_array_equal(x[1], [0, 7, 0, 5])
        self.assertTrue(np.isnan(x[2]).all())


if __name__ == "__main__":
    unittest.main()
