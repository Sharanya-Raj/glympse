"""Acceptance tests for leakage, interpolation, group isolation and endpoints."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from meal_response.data import Patient, load_sources, medication_events
from meal_response.evaluation import split_manifest, training_weights
from meal_response.features import TARGETS, build_examples, forecast, label_curve, make_features, summarize
from meal_response.training import predict_bundle, train_run
from meal_response.__main__ import input_patient


def patient(source="t2dm"):
    time = pd.Timestamp("2025-01-10 12:00")
    times = pd.date_range(time - pd.Timedelta(days=8), time + pd.Timedelta(minutes=130), freq="5min")
    glucose = pd.DataFrame({"time": times, "glucose": 100 + np.arange(len(times)) % 10,
                            "interpolated": False})
    meals = pd.DataFrame([{"time": time, "carbs": 40.0, "protein": 20.0, "fat": 10.0,
                           "calories": 330.0, "fiber": np.nan, "nutrition_complete": True,
                           "meal_id": "one", "recording": "rec"}])
    return Patient("person", source, "t2dm", glucose, meals,
                   pd.DataFrame(columns=["time", "mets", "hr", "activity_calories"]),
                   pd.DataFrame(columns=["time", "drug", "route", "unit", "dose", "raw", "parsed"]))


class CausalityTests(unittest.TestCase):
    def test_future_and_old_values_do_not_change_features(self):
        for source in ["cg_macros", "t2dm"]:
            p = patient(source)
            t = p.meals.time.iloc[0]
            before = make_features(p, t, p.meals.iloc[0])
            p.glucose.loc[(p.glucose.time > t) | (p.glucose.time < t - pd.Timedelta(days=7)), "glucose"] = 999
            after = make_features(p, t, p.meals.iloc[0])
            pd.testing.assert_series_equal(pd.Series(before), pd.Series(after))

    def test_t2dm_supplied_interpolation_never_enters_inputs(self):
        p = patient()
        t = p.meals.time.iloc[0]
        p.glucose["interpolated"] = np.arange(len(p.glucose)) % 3 != np.flatnonzero(p.glucose.time.eq(t))[0] % 3
        before = make_features(p, t, p.meals.iloc[0])
        p.glucose.loc[p.glucose.interpolated, "glucose"] += 1000
        pd.testing.assert_series_equal(pd.Series(before), pd.Series(make_features(p, t, p.meals.iloc[0])))

    def test_cg_guard_excludes_latest_dense_minutes(self):
        p = patient("cg_macros")
        t = p.meals.time.iloc[0]
        before = make_features(p, t, p.meals.iloc[0])
        p.glucose.loc[p.glucose.time > t - pd.Timedelta(minutes=5), "glucose"] = 999
        pd.testing.assert_series_equal(pd.Series(before), pd.Series(make_features(p, t, p.meals.iloc[0])))

    def test_short_history_allowed_and_stale_baseline_rejected(self):
        p = patient()
        t = p.meals.time.iloc[0]
        p.glucose = p.glucose.loc[p.glucose.time >= t - pd.Timedelta(minutes=60)]
        self.assertLess(make_features(p, t, p.meals.iloc[0])["history_days"], 1)
        p.glucose = p.glucose.loc[p.glucose.time < t - pd.Timedelta(minutes=5)]
        with self.assertRaisesRegex(ValueError, "stale_baseline"):
            make_features(p, t, p.meals.iloc[0])

    def test_missing_nutrition_still_excludes_overlap(self):
        p = patient()
        second = p.meals.iloc[0].copy()
        second.time += pd.Timedelta(minutes=60)
        second.nutrition_complete = False
        p.meals = pd.concat([p.meals, second.to_frame().T], ignore_index=True)
        frame, audit = build_examples([p])
        self.assertTrue(frame.empty)
        self.assertEqual(audit[0]["overlapping_meal"], 1)

    def test_missing_label_is_not_interpolated(self):
        p = patient()
        t = p.meals.time.iloc[0]
        p.glucose = p.glucose.loc[p.glucose.time.ne(t + pd.Timedelta(minutes=30))]
        with self.assertRaisesRegex(ValueError, "incomplete_curve"):
            label_curve(p, t)

    def test_observed_policy_refuses_unknown_cg_provenance(self):
        p = patient("cg_macros")
        with self.assertRaisesRegex(ValueError, "unknown_observation_provenance"):
            label_curve(p, p.meals.time.iloc[0], observed_only=True)

    def test_negative_macros_rejected(self):
        p = patient()
        meal = p.meals.iloc[0].copy()
        meal.carbs = -1
        with self.assertRaisesRegex(ValueError, "invalid_nutrition"):
            make_features(p, meal.time, meal)
        meal.carbs = 40
        meal.fiber = None
        self.assertTrue(np.isnan(make_features(p, meal.time, meal)["fiber"]))
        meal.fiber = -1
        with self.assertRaisesRegex(ValueError, "invalid_nutrition"):
            make_features(p, meal.time, meal)

    def test_recordings_do_not_supply_each_others_labels(self):
        p = patient()
        t = p.meals.time.iloc[0]
        p.glucose["recording"] = "first"
        p.glucose.loc[p.glucose.time > t + pd.Timedelta(minutes=30), "recording"] = "second"
        with self.assertRaisesRegex(ValueError, "incomplete_curve"):
            label_curve(p, t, recording="first")

    def test_enhanced_histories_are_bounded(self):
        t = patient().meals.time.iloc[0]
        for source in ["t2dm", "cg_macros"]:
            p = patient(source)
            times = [t - pd.Timedelta(days=8), t - pd.Timedelta(minutes=20), t + pd.Timedelta(minutes=5)]
            p.activity = pd.DataFrame({"time": times, "mets": [1.0, 2.0, 3.0], "hr": [60.0, 70.0, 80.0], "activity_calories": [1.0, 2.0, 3.0]})
            p.medication = pd.DataFrame({"time": times, "drug": "metformin", "route": "oral", "unit": "mg", "dose": [100, 200, 300], "raw": "record", "parsed": True})
            before = make_features(p, t, p.meals.iloc[0], enhanced=True)
            p.activity.loc[[0, 2], ["mets", "hr", "activity_calories"]] = 999
            p.medication.loc[[0, 2], "dose"] = 999
            pd.testing.assert_series_equal(pd.Series(before), pd.Series(make_features(p, t, p.meals.iloc[0], enhanced=True)))


class OutcomeTests(unittest.TestCase):
    def test_constant_above_baseline_curve(self):
        result = summarize(np.full((1, 24), 110), [100]).iloc[0]
        self.assertEqual(result.peak, 110)
        self.assertEqual(result.peak_rise, 10)
        self.assertEqual(result.time_to_peak, 5)
        self.assertEqual(result.iauc, 1175)
        self.assertEqual(result.net_iauc, 1175)
        self.assertEqual(result.glucose_120, 110)

    def test_below_baseline_and_linear_rise(self):
        low = summarize(np.full((1, 24), 90), [100]).iloc[0]
        self.assertEqual(low.iauc, 0)
        self.assertEqual(low.net_iauc, -1175)
        self.assertEqual(low.peak_rise, -10)
        rise = summarize(100 + np.arange(5, 121, 5), [100]).iloc[0]
        self.assertEqual(rise.iauc, 7200)
        self.assertEqual(rise.time_to_peak, 120)

    def test_medication_identity_and_units_preserved(self):
        frame = pd.DataFrame({"time": [pd.Timestamp("2025-01-01")],
                              "Non-insulin hypoglycemic agents": ["metformin 0.5 g, acarbose 50 mg"],
                              "Insulin dose - s.c.": ["insulin aspart 70/30, 14 IU"]})
        events = medication_events(frame)
        self.assertEqual(set(events.drug), {"metformin", "acarbose", "insulin aspart 70/30"})
        self.assertEqual(events.loc[events.drug.eq("metformin"), "unit"].iloc[0], "g")


class TrainingTests(unittest.TestCase):
    def examples(self):
        rows = []
        for source in ["cg_macros", "t2dm"]:
            for i in range(10):
                for j in range(2):
                    rows.append({"participant": f"{source}:{i}", "source": source, "cohort": "t2dm",
                                 "time": pd.Timestamp("2025-01-01") + pd.Timedelta(days=j),
                                 "meal_id": f"{i}:{j}", "baseline": 100.0, "history_days": 1.0,
                                 "supplied_label_fraction": 0.0, "carbs": float(i + j),
                                 **{name: 100 + i + j + h for h, name in enumerate(TARGETS)}})
        return pd.DataFrame(rows)

    def test_patient_split_stable_and_disjoint(self):
        examples = self.examples()
        a = split_manifest(examples)
        b = split_manifest(examples.sample(frac=1, random_state=1))
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.participant.duplicated().any())
        self.assertEqual(set(a.partition), {"train", "validation", "test"})

    def test_weights_equal_sources_and_patients(self):
        examples = self.examples()
        examples = pd.concat([examples, examples.iloc[[0, 0, 0]]], ignore_index=True)
        examples["weight"] = training_weights(examples)
        self.assertEqual(examples.groupby("source").weight.sum().nunique(), 1)
        totals = examples.groupby("participant").weight.sum().to_numpy()
        np.testing.assert_allclose(totals, totals[0])

    def test_both_families_and_all_dataset_options_reload(self):
        examples = self.examples()
        manifest = split_manifest(examples)
        with tempfile.TemporaryDirectory() as tmp:
            for dataset in ["cg_macros", "t2dm", "combined"]:
                for family in ["linear", "xgboost"]:
                    output = Path(tmp) / f"{dataset}_{family}"
                    report = train_run(examples, manifest, dataset, family, "core", output,
                                       trials=1, max_trees=5, bootstraps=2)
                    self.assertEqual(report["dataset"], dataset)
                    bundle = joblib.load(output / "model.joblib")
                    self.assertNotIn("macro_confidence", bundle["feature_names"])
                    self.assertTrue((output / "training_weights.csv").exists())
                    self.assertEqual(predict_bundle(bundle, examples).shape, (40, 24))

    def test_inference_reuses_training_features_and_handles_omitted_histories(self):
        p = patient()
        time = p.meals.time.iloc[0]
        p.glucose = p.glucose.loc[p.glucose.time <= time]
        payload = {"source": "t2dm", "glucose_history": p.glucose.assign(time=p.glucose.time.astype(str)).to_dict(orient="records")}
        parsed = input_patient(payload)
        expected = make_features(p, time, p.meals.iloc[0])
        actual = make_features(parsed, time, p.meals.iloc[0])
        pd.testing.assert_series_equal(pd.Series(expected), pd.Series(actual))

    def test_json_null_fiber_history_is_supported(self):
        p = patient()
        time = p.meals.time.iloc[0]
        payload = {"source": "t2dm", "glucose_history": p.glucose.assign(time=p.glucose.time.astype(str)).to_dict(orient="records"),
                   "meal_history": [{"time": str(time - pd.Timedelta(minutes=180)), "carbs": 40, "protein": 20,
                                     "fat": 10, "calories": 330, "fiber": None, "nutrition_complete": True}]}
        features = make_features(input_patient(payload), time, p.meals.iloc[0])
        self.assertTrue(np.isnan(features["history_1d_logged_fiber"]))
        self.assertEqual(features["history_1d_logged_carbs"], 40)

    def test_missing_unselected_source_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t2dm.csv"
            pd.DataFrame({"Patient": ["one"], "Recording": ["visit"], "Date": ["2025-01-01"],
                          "CGM (mg / dl)": [100], "Interpolated": [False], "meal_id": ["meal"],
                          "Dietary intake": ["rice"], "macro_status": ["ok"], "meal_carbs_g": [40],
                          "meal_protein_g": [10], "meal_fat_g": [5], "meal_calories_kcal": [245]}).to_csv(path, index=False)
            patients, metadata = load_sources(path, Path(tmp) / "missing")
            self.assertEqual([p.source for p in patients], ["t2dm"])
            self.assertEqual(len(metadata["files"]), 1)


if __name__ == "__main__":
    unittest.main()
