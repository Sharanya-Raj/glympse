"""Small generated fixtures verify alignment and exclusion, not clinical accuracy."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_participants
from .features import build_meals


class PipelineTests(unittest.TestCase):
    def fixture(self):
        times = pd.date_range("2025-01-01", periods=10 * 288, freq="5min")
        frame = pd.DataFrame({"Timestamp": times, "Dexcom GL": 100.0,
                              "Meal Type": "", "Carbs": np.nan,
                              "Protein": np.nan, "Fat": np.nan, "Fiber": np.nan})
        for day in range(10):
            for hour in (8, 13, 18):
                index = day * 288 + hour * 12
                frame.loc[index, ["Meal Type", "Carbs", "Protein", "Fat", "Fiber"]] = ["Breakfast", 40, 20, 10, 5]
                frame.loc[index:index + 24, "Dexcom GL"] = np.linspace(100, 124, 25)
        with tempfile.TemporaryDirectory() as directory:
            frame.to_csv(Path(directory) / "CGMacros-001.csv", index=False)
            return load_participants(directory)

    def test_horizon_alignment_and_no_future_features(self):
        data = self.fixture()
        meals = build_meals(data)
        self.assertEqual(len(meals), 30)
        self.assertEqual(meals.iloc[0].glucose_30, 106)
        self.assertEqual(meals.iloc[0].glucose_120, 124)
        self.assertEqual(meals.iloc[0].time_to_peak, 120)
        self.assertEqual(len(meals[meals.day == 8]), 3)
        cutoff = meals.iloc[0].Timestamp
        changed = data.copy()
        changed.loc[changed.Timestamp > cutoff, "Dexcom GL"] += 50
        altered = build_meals(changed)
        self.assertEqual(meals.iloc[0].history_mean, altered.iloc[0].history_mean)
        self.assertEqual(meals.iloc[0].baseline, altered.iloc[0].baseline)

    def test_missing_target_and_overlapping_meal_are_excluded(self):
        data = self.fixture()
        start = pd.Timestamp("2025-01-01 08:00")
        data.loc[data.Timestamp.between(start + pd.Timedelta(minutes=20),
                                       start + pd.Timedelta(minutes=50)), "Dexcom GL"] = np.nan
        self.assertEqual(len(build_meals(data)), 29)
        data = self.fixture()
        data.loc[data.Timestamp == start + pd.Timedelta(minutes=60), "Meal Type"] = "Snack"
        self.assertFalse((build_meals(data).Timestamp == start).any())

    def test_train_and_save(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("legacy_model", Path(__file__).parent / "regression-ml-glucose/legacy_cgmacros_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train = module.train
        from .explain import main as explain
        with tempfile.TemporaryDirectory() as directory:
            meals = build_meals(self.fixture())
            report = train(meals, directory)
            self.assertEqual(report["train_meals"], 21)
            self.assertEqual(report["test_meals"], 3)
            self.assertTrue((Path(directory) / "regression.joblib").exists())
            meals.to_csv(Path(directory) / "meals.csv", index=False)
            with patch("sys.argv", ["explain", "--model", str(Path(directory) / "regression.joblib"),
                                   "--meals", str(Path(directory) / "meals.csv"),
                                   "--output", directory]):
                explain()
            self.assertTrue((Path(directory) / "importance_glucose_60.csv").exists())


if __name__ == "__main__":
    unittest.main()
