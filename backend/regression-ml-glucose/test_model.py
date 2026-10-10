import unittest

import numpy as np
import pandas as pd

from model import make_windows
from normalize import enrich
from prepare_data import prepare


class RegressionTests(unittest.TestCase):
    def test_text_parsing(self):
        frame = pd.DataFrame({"Patient": ["1", "1"], "Recording": ["a", "a"],
            "Date": ["2020-01-01 08:00", "2020-01-01 08:05"],
            "Dietary intake": ["Rice 100 g\\nMilk 250 ml", "No food ingested"],
            "Insulin dose - s.c.": ["insulin glargine, 14 IU, Humulin 70/30, 12 IU", None],
            "Non-insulin hypoglycemic agents": ["metformin 0.5 g, acarbose 50 mg", "voglibose 0.2 g"],
            "CSII - bolus insulin (Novolin R, IU)": ["acarbose 50 mg", None],
            "CSII - basal insulin (Novolin R, IU / H)": ["0.5", None]})
        result = enrich(prepare(frame))
        self.assertEqual(result.iloc[0].food_amount_grams, 100)
        self.assertEqual(result.iloc[0].food_amount_ml, 250)
        self.assertEqual(result.iloc[0].insulin_sc_iu, 26)
        self.assertEqual(result.iloc[0].med_metformin_mg, 500)
        self.assertEqual(result.iloc[0].med_acarbose_mg, 50)
        self.assertTrue(pd.isna(result.iloc[1].med_voglibose_mg))
        self.assertEqual(result.iloc[0].csii_bolus_iu_unparsed, 1)

    def test_future_never_enters_features(self):
        dates = pd.date_range("2020-01-01 13:54", periods=9 * 288, freq="5min")
        data = pd.DataFrame({"Patient": "1", "Recording": "a", "Date": dates,
                             "CGM (mg / dl)": np.repeat(np.arange(9) + 100, 288),
                             "food_logged_now": 0})
        first, _ = make_windows(data)
        data.loc[data.Date >= dates[7 * 288], "CGM (mg / dl)"] = 300
        data.loc[data.Date >= dates[7 * 288], "food_logged_now"] = 1
        changed, _ = make_windows(data)
        self.assertEqual(first.iloc[0].target_day, 8)
        self.assertEqual(first.iloc[0].day7_glucose_mean, 106)
        self.assertEqual(first.iloc[0].target, 107)
        for column in first.columns.difference(["target"]):
            self.assertEqual(first.iloc[0][column], changed.iloc[0][column])


if __name__ == "__main__":
    unittest.main()
