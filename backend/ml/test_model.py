import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from .model import (CONTEXT, TARGETS, fit_personal, fit_population,
                                meal_examples, predict, personal_history)
from .benchmarks import benchmark, scores, export_daily_comparisons
from .normalize import enrich
from .prepare import prepare
from .explain import daily_top_three, contributions


class PersonalizedTests(unittest.TestCase):
    def test_daily_rank_uses_magnitude_and_patient_isolation(self):
        records = []
        for horizon in [30, 60]:
            for feature, value in [("food_starch", 10 if horizon == 30 else -10),
                                   ("baseline", 5), ("insulin_sc_iu", -3), ("food_protein", 1)]:
                records.append({"Patient": "alice", "Recording": "a", "day": 8,
                                "feature": feature, "contribution": value})
        records.append({"Patient": "bob", "Recording": "a", "day": 8,
                        "feature": "other", "contribution": 100})
        ranked = daily_top_three(pd.DataFrame(records))
        alice = ranked[ranked.Patient == "alice"]
        self.assertEqual(alice.feature.tolist(), ["food_starch", "baseline", "insulin_sc_iu"])
        self.assertEqual(alice.iloc[0].importance_mg_dl, 10)
        self.assertEqual(alice.iloc[0].direction, "balanced")

    def test_percentage_error_and_tolerance_accuracy(self):
        frame = pd.DataFrame({"Patient": ["1"] * 3, "actual_glucose": [100., 200., 0.],
            "predicted_glucose": [110., 240., 10.], "baseline_prediction": [100.] * 3})
        result = scores(frame, "predicted_glucose")
        self.assertAlmostEqual(result["mape_percent"], 15.)
        self.assertEqual(result["percentage_metric_count"], 2)
        self.assertEqual(result["within_15_percent_accuracy"], 50.)
        self.assertAlmostEqual(result["within_20_mg_dl_percent"], 200 / 3)

    def test_benchmark_comparators(self):
        frame = pd.DataFrame({"Patient": ["1", "1", "2", "2"], "day": [8, 9, 8, 9],
            "Recording": "a", "meal_time": ["2020-01-08", "2020-01-09"] * 2,
            "horizon_minutes": 30, "actual_glucose": [110., 120., 130., 140.],
            "predicted_glucose": [110., 120., 130., 140.],
            "population_prediction": [100., 110., 120., 130.],
            "baseline_prediction": 100., "personal_mean_prediction": 115.})
        perfect = scores(frame, "predicted_glucose")
        self.assertEqual(perfect["mae"], 0)
        self.assertEqual(perfect["r2"], 1)
        self.assertEqual(perfect["change_r2"], 1)
        with patch.object(pd.DataFrame, "to_csv"), patch.object(Path, "write_text"), patch.object(Path, "mkdir"):
            report = benchmark(frame, Path("unused"))
        self.assertEqual(len(report), 4)
        row = report[report.model == "personalized"].iloc[0]
        self.assertEqual(row.macro_mae_ci_low, 0)
        self.assertEqual(row.macro_mae_ci_high, 0)

    def test_daily_comparisons_isolate_patient_recording_and_day(self):
        frame = pd.DataFrame({"Patient": ["1", "1", "2", "1"],
            "Recording": ["a", "a", "a", "b"], "day": [8, 9, 8, 8],
            "meal_time": ["2020-01-08", "2020-01-09", "2020-01-08", "2020-02-08"],
            "horizon_minutes": 30, "actual_glucose": 100.,
            "predicted_glucose": [110., 130., 140., 100.], "baseline_prediction": 100.})
        with patch.object(pd.DataFrame, "to_csv") as save:
            comparison, daily = export_daily_comparisons(frame, Path("unused"))
        self.assertEqual(save.call_count, 3)
        self.assertEqual(len(comparison), 3)
        self.assertEqual(comparison.day.unique().tolist(), [8])
        self.assertEqual(comparison.absolute_error_mg_dl.tolist(), [10., 0., 40.])
        totals = daily[daily.horizon_minutes == "all"]
        self.assertEqual(len(totals), 4)
        self.assertEqual(totals.within_20_mg_dl_percent.tolist(), [100., 0., 100., 0.])
        self.assertTrue(totals.r2.isna().all())

    def test_food_and_medication_parsing(self):
        frame = pd.DataFrame({"Patient": ["1"], "Recording": ["a"], "Date": ["2020-01-01"],
            "Dietary intake": ["Rice 100 g Milk 250 ml"],
            "Insulin dose - s.c.": ["insulin glargine, 14 IU, Humulin 70/30, 12 IU"],
            "Non-insulin hypoglycemic agents": ["metformin 0.5 g, acarbose 50 mg"],
            "CSII - bolus insulin (Novolin R, IU)": ["acarbose 50 mg"],
            "CSII - basal insulin (Novolin R, IU / H)": ["0.5"]})
        result = enrich(prepare(frame)).iloc[0]
        self.assertEqual(result.food_amount_grams, 100)
        self.assertEqual(result.food_amount_ml, 250)
        self.assertEqual(result.insulin_sc_iu, 26)
        self.assertEqual(result.med_metformin_mg, 500)
        self.assertEqual(result.csii_bolus_iu_unparsed, 1)

    def test_rolling_personal_cutoffs(self):
        times = pd.date_range("2020-01-01 08:00", periods=10, freq="D")
        frame = pd.DataFrame({"Recording": "a", "meal_time": times,
                              "label_end": times + pd.Timedelta(hours=2)})
        day8 = personal_history(frame, "a", pd.Timestamp("2020-01-08"))
        day9 = personal_history(frame, "a", pd.Timestamp("2020-01-09"))
        self.assertEqual(day8.meal_time.dt.day.tolist(), list(range(1, 8)))
        self.assertEqual(day9.meal_time.dt.day.tolist(), list(range(2, 9)))
        self.assertTrue(personal_history(frame, "other", pd.Timestamp("2020-01-08")).empty)

    def test_meal_features_do_not_use_future(self):
        times = pd.date_range("2020-01-01", periods=9 * 288, freq="5min")
        data = pd.DataFrame({"Patient": "1", "Recording": "a", "Date": times,
                             "CGM (mg / dl)": 100., "food_logged_now": 0,
                             "food_starch": 0, "insulin_sc_iu": 0.})
        for day in range(9):
            data.loc[day * 288 + 96, ["food_logged_now", "food_starch"]] = 1
        meals, features, _ = meal_examples(data)
        self.assertEqual(meals.day.tolist(), list(range(1, 10)))
        changed = data.copy()
        cutoff = pd.Timestamp("2020-01-08 00:00")
        changed.loc[changed.Date >= cutoff, "CGM (mg / dl)"] = 200
        other, _, _ = meal_examples(changed)
        pd.testing.assert_frame_equal(meals[meals.day < 8], other[other.day < 8])
        self.assertEqual(meals.iloc[0].baseline, 100)
        self.assertEqual(meals.iloc[0].delta_120, 0)
        self.assertNotIn("Patient", features)

    def test_population_excludes_patient_and_personalization_is_local(self):
        rows = []
        for patient in ["1", "2", "3"]:
            for i in range(15):
                rows.append({"Patient": patient, "recording_start": pd.Timestamp("2020-01-01"),
                    "label_end": pd.Timestamp("2020-01-02"),
                    **{feature: float(i) for feature in CONTEXT},
                    **{target: 20. if patient == "1" else 0. for target in TARGETS}})
        examples = pd.DataFrame(rows)
        model, ids = fit_population(examples, CONTEXT, "1")
        self.assertNotIn("1", ids)
        history = examples[examples.Patient == "1"]
        coef, intercept = fit_personal(model, history, CONTEXT)
        adapted = predict(model, coef, intercept, history[CONTEXT])
        effects, reference = contributions(model, coef, CONTEXT, history[CONTEXT])
        reconstructed = reference + model[-1].intercept_ + intercept + effects.sum(axis=2)
        np.testing.assert_allclose(reconstructed, adapted + history.baseline.to_numpy()[:, None])
        self.assertGreater(adapted.mean(), model.predict(history[CONTEXT]).mean())
        coef, intercept = fit_personal(model, history.head(4), CONTEXT)
        np.testing.assert_equal(coef, 0)
        np.testing.assert_equal(intercept, 0)


class ExportTests(unittest.TestCase):
    def test_fifteen_minute_cgm_fills_short_gaps_only(self):
        from .export_patient import to_five_minute
        times = list(pd.date_range("2021-01-01 00:00", periods=4, freq="15min")) + [pd.Timestamp("2021-01-01 02:00")]
        raw = pd.DataFrame({"Date": times, "CGM (mg / dl)": [100, 115, 130, 145, 160],
                            "Dietary intake": [None, "Rice 100 g", None, None, None]})
        out = to_five_minute(raw).set_index("Date")
        self.assertAlmostEqual(out.loc["2021-01-01 00:05", "CGM (mg / dl)"], 105)
        self.assertAlmostEqual(out.loc["2021-01-01 00:40", "CGM (mg / dl)"], 140)
        # 00:45 -> 02:00 is a 75-minute gap: no invented readings inside it.
        self.assertTrue(out.loc["2021-01-01 00:50":"2021-01-01 01:55", "CGM (mg / dl)"].isna().all())
        self.assertEqual(out.loc["2021-01-01 02:00", "CGM (mg / dl)"], 160)
        # Meal text stays on its own timestamp only.
        self.assertEqual(out["Dietary intake"].notna().sum(), 1)
        self.assertEqual(out.loc["2021-01-01 00:15", "Dietary intake"], "Rice 100 g")


if __name__ == "__main__":
    unittest.main()
