import unittest
import pandas as pd
from .prediction_summary import evidence_summary, explain_peak, summarize_prediction
from .test_api import fixture
from .ml.service import validate_history


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.raw = validate_history(fixture())
        self.result = {"personal_cutoff": "2025-01-08T00:00:00", "meal_time": "2025-01-08T08:00:00", "personal_training_meals": 10,
            "predictions": [{"minutes_after_meal": 30, "glucose_mg_dl": 140}, {"minutes_after_meal": 60, "glucose_mg_dl": 150}],
            "feature_contributions": [{"feature": name, "contribution": value, "horizon_minutes": 60} for name,value in
                [("hour_sin", 8), ("hour_cos", -6), ("baseline", 10), ("food_starch", 5), ("food_sweet", 4), ("med_metformin", -3)]]}
        self.recent = [{"timestamp": t.isoformat(), "glucose": 110} for t in pd.date_range("2025-01-08 07:30", periods=7, freq="5min")]

    def test_daily_coverage_and_no_false_missing_meals(self):
        summary = evidence_summary(self.result, self.raw, self.recent, "Rice 100 g")
        self.assertEqual(summary["gap_count"], 1) # medication completeness cannot be established
        self.assertEqual(summary["cgm_coverage_percent"], 100)
        self.assertEqual([d["recorded_meal_events"] for d in summary["daily"]], [3]*7)
        self.assertFalse(summary["calibrated"])

    def test_empty_day_future_rows_and_slot_deduplication(self):
        raw = self.raw[~self.raw.Date.between("2025-01-03", "2025-01-03 23:59:59")].copy()
        result = evidence_summary(self.result, raw, self.recent[:4], "Rice")
        self.assertEqual(result["daily"][2]["cgm_slots_present"], 0)
        self.assertEqual(result["gap_count"], 4)
        changed = raw.copy()
        changed.loc[changed.Date >= pd.Timestamp("2025-01-08"), "CGM (mg / dl)"] = 500
        self.assertEqual(result, evidence_summary(self.result, changed, self.recent[:4], "Rice"))

    def test_grouped_signed_contributions_and_no_mutation(self):
        explanation = explain_peak(self.result)
        self.assertEqual([f["label"] for f in explanation["top_factors"]], ["Baseline glucose", "Meal type", "Medications"])
        self.assertEqual(explanation["top_factors"][1]["contribution_mg_dl"], 9)
        augmented = summarize_prediction(self.result, self.raw, self.recent, "Rice 100 g")
        self.assertEqual(augmented["predictions"], self.result["predictions"])
        self.assertNotIn("uncertainty", self.result)
        self.result["feature_contributions"] = []
        self.assertEqual(explain_peak(self.result)["top_factors"], [])
