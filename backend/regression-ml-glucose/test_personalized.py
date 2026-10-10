import unittest

import numpy as np
import pandas as pd

from personalized_model import (CONTEXT, TARGETS, fit_personal, fit_population,
                                meal_examples, predict, personal_history)


class PersonalizedTests(unittest.TestCase):
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
        self.assertGreater(adapted.mean(), model.predict(history[CONTEXT]).mean())
        coef, intercept = fit_personal(model, history.head(4), CONTEXT)
        np.testing.assert_equal(coef, 0)
        np.testing.assert_equal(intercept, 0)


if __name__ == "__main__":
    unittest.main()
