import unittest
import tempfile
from pathlib import Path

import joblib

import numpy as np
import pandas as pd

from meal_response.features import TARGETS
from meal_response.weighting import confidence_factors, loss_weights


def examples(confidence=(1.0, 0.9, 0.45, 0.0), source="t2dm"):
    return pd.DataFrame({"participant": "person", "source": source, "meal_id": [f"m{i}" for i in range(4)],
                         "time": pd.date_range("2025-01-01", periods=4), "baseline": 100.0,
                         "macro_confidence": np.asarray(confidence, dtype=float),
                         **{name: [110.0, 120.0, 130.0, 180.0] for name in TARGETS}})


class WeightingTests(unittest.TestCase):
    def test_high_confidence_normal_low_proportional_zero_excluded(self):
        np.testing.assert_allclose(confidence_factors(examples()), [1, 1, 0.5, 0])
        weights, details, policy = loss_weights(examples())
        self.assertEqual(weights[0], weights[1])
        self.assertAlmostEqual(weights[2] / weights[0], 0.5)
        self.assertEqual(weights[3], 0)
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertEqual(policy["confidence_downweighted_meals"], 2)

    def test_unscored_sources_are_explicit(self):
        frame = examples([np.nan] * 4)
        np.testing.assert_allclose(confidence_factors(frame), 0.5)
        frame.source = "cg_macros"
        np.testing.assert_allclose(confidence_factors(frame), 1)
        _, detail, _ = loss_weights(frame)
        self.assertTrue(detail.confidence_origin.eq("unscored_normal_weight").all())

    def test_bad_confidence_and_all_zero_fail(self):
        for value in [-0.1, 1.1, np.inf]:
            with self.assertRaises(ValueError):
                confidence_factors(examples([value] * 4))
        with self.assertRaisesRegex(ValueError, "No positive"):
            loss_weights(examples([0.0] * 4))

    def test_training_rise_emphasis_and_validation_isolation(self):
        frame = examples([1.0] * 4)
        weights, detail, policy = loss_weights(frame, large_rise_multiplier=1.25, emphasize_rises=True)
        self.assertAlmostEqual(weights[3] / weights[0], 1.25)
        self.assertEqual(policy["large_rise_upweighted_meals"], 1)
        before, _, policy = loss_weights(frame, large_rise_multiplier=1.25, emphasize_rises=False)
        frame.loc[3, TARGETS] = 10000
        after, _, _ = loss_weights(frame, large_rise_multiplier=1.25, emphasize_rises=False)
        np.testing.assert_allclose(before, after)
        self.assertFalse(policy["rise_emphasis_applied"])

    def test_quality_still_reduces_weight_after_rise_boost(self):
        frame = examples([1, 1, 1, 0.45])
        weights, _, _ = loss_weights(frame, large_rise_multiplier=1.25, emphasize_rises=True)
        self.assertAlmostEqual(weights[3] / weights[0], 0.625)

    def test_macro_confidence_is_not_an_inference_feature(self):
        from meal_response.features import META
        self.assertIn("macro_confidence", META)

    def test_held_out_meal_changes_cannot_change_fit_or_training_weights(self):
        from meal_response.evaluation import split_manifest
        from meal_response.training import train_run, predict_bundle
        frame = pd.concat([examples([1] * 4).assign(participant=f"person{i}", cohort="t2dm",
                           history_days=1.0, supplied_label_fraction=0.0) for i in range(6)], ignore_index=True)
        manifest = split_manifest(frame)
        changed = frame.copy()
        mask = changed.participant.isin(manifest.loc[manifest.partition.eq("test"), "participant"])
        changed.loc[mask, TARGETS] = 10000
        changed.loc[mask, "macro_confidence"] = 0.1
        with tempfile.TemporaryDirectory() as tmp:
            outputs = [Path(tmp) / "before", Path(tmp) / "after"]
            for data, output in zip([frame, changed], outputs):
                train_run(data, manifest, "t2dm", "linear", "core", output, bootstraps=0)
            before = joblib.load(outputs[0] / "model.joblib")
            after = joblib.load(outputs[1] / "model.joblib")
            self.assertEqual(before["weighting"], after["weighting"])
            np.testing.assert_allclose(predict_bundle(before, frame), predict_bundle(after, frame))
            pd.testing.assert_frame_equal(pd.read_csv(outputs[0] / "training_weights.csv"),
                                          pd.read_csv(outputs[1] / "training_weights.csv"))


if __name__ == "__main__":
    unittest.main()
