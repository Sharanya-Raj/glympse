import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx
import joblib
import numpy as np
import pandas as pd

from .app import create_app
from .ml.model import fit_population, meal_examples
from .ml.service import COLUMNS, prepared_history, validate_history, forecast_meal


class ApiClient:
    """ASGI transport works with the existing FastAPI/httpx versions."""
    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kwargs):
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(send())

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)


def fixture(patient="1", days=9):
    timestamps = pd.date_range("2025-01-01", periods=days * 288, freq="5min")
    frame = pd.DataFrame({"Date": timestamps, "CGM (mg / dl)": 110.0,
                          "Patient": patient, "Recording": "a", "Dietary intake": ""})
    for day in range(days):
        for hour in (8, 13, 18):
            i = day * 288 + hour * 12
            frame.loc[i, "Dietary intake"] = "Rice 100 g Egg 50 g"
            frame.loc[i:i + 24, "CGM (mg / dl)"] = np.linspace(110, 145, 25)
    for col in COLUMNS:
        if col not in frame:
            frame[col] = pd.NA
    return frame[COLUMNS]


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cohort = pd.concat([fixture("other1", 7), fixture("other2", 7)], ignore_index=True)
        examples, features, _ = meal_examples(prepared_history(cohort))
        model, _ = fit_population(examples, features, None)
        cls.bundle = {"model": model, "features": features, "schema": 1, "version": "test"}

    def test_signup_upload_predict_daily_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            artifact = path / "population.joblib"
            joblib.dump(self.bundle, artifact)
            client = ApiClient(create_app(path, artifact))
            first = client.post("/accounts", json={"username": "alice", "password": "test-password-123"})
            self.assertEqual(first.status_code, 201)
            headers = {"Authorization": "Bearer " + first.json()["access_token"]}
            second = client.post("/accounts", json={"username": "bob", "password": "test-password-456"})
            other_headers = {"Authorization": "Bearer " + second.json()["access_token"]}
            self.assertEqual(client.get("/history", headers=other_headers).status_code, 409)
            self.assertEqual(client.get("/history").status_code, 401)
            raw = fixture(days=7)
            uploaded = client.post("/history", headers=headers,
                files={"file": ("history.csv", raw.to_csv(index=False), "text/csv")})
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            self.assertTrue(uploaded.json()["ready_for_day8"])
            request = {"meal_time": "2025-01-08T08:00:00Z", "dietary_intake": "Rice 100 g Egg 50 g",
                "recent_cgm": [{"timestamp": str(time), "glucose": 110 + i}
                    for i, time in enumerate(pd.date_range("2025-01-08 07:30", periods=7, freq="5min"))]}
            prediction = client.post("/predict", headers=headers, json=request)
            self.assertEqual(prediction.status_code, 200, prediction.text)
            self.assertTrue(prediction.json()["personalized"])
            self.assertEqual(len(prediction.json()["predictions"]), 4)
            explanation = client.get("/explanations/daily", headers=headers).json()
            self.assertTrue(1 <= len(explanation["days"][0]["top_features"]) <= 3)
            self.assertEqual(client.get("/explanations/daily", headers=other_headers).json()["days"], [])
            # Retry replaces the same meal forecast instead of inflating its ranking weight.
            client.post("/predict", headers=headers, json=request)
            retry = client.get("/explanations/daily", headers=headers).json()
            self.assertEqual(explanation, retry)
            # History change invalidates old rankings.
            client.post("/history", headers=headers,
                files={"file": ("history.csv", raw.to_csv(index=False), "text/csv")})
            self.assertEqual(client.get("/explanations/daily", headers=headers).json()["days"], [])

    def test_no_future_outcomes_and_roll_day9(self):
        raw = validate_history(fixture())
        request = [{"timestamp": str(t), "glucose": 110}
                   for t in pd.date_range("2025-01-08 07:30", periods=7, freq="5min")]
        original = forecast_meal(raw, "2025-01-08 08:00", "Rice 100 g", request, self.bundle)
        altered = raw.copy()
        altered.loc[altered.Date >= pd.Timestamp("2025-01-08 08:05"), "CGM (mg / dl)"] = 500
        self.assertEqual(original, forecast_meal(altered, "2025-01-08 08:00", "Rice 100 g", request, self.bundle))
        request = [{"timestamp": str(t), "glucose": 110}
                   for t in pd.date_range("2025-01-09 07:30", periods=7, freq="5min")]
        self.assertEqual(forecast_meal(raw, "2025-01-09 08:00", "Rice 100 g", request, self.bundle)["day"], 9)


if __name__ == "__main__":
    unittest.main()
