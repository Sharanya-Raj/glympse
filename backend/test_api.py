import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx
import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError
from unittest.mock import patch
import os
import boto3
from moto import mock_aws
from fastapi import HTTPException
from .accounts import Accounts

from .app import create_app
from .ml.model import fit_population, meal_examples
from .ml.service import COLUMNS, prepared_history, validate_history, forecast_meal
from .ml.spike import variants


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
        # Exercise the existing artifact; never train or replace it in API tests.
        cls.bundle = joblib.load(Path(__file__).parent / "artifacts/population.joblib")

    def setUp(self):
        env = patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-west-2"})
        env.start(); self.addCleanup(env.stop)
        aws = mock_aws(); aws.start(); self.addCleanup(aws.stop)
        resource = boto3.resource("dynamodb", region_name="us-west-2")
        for name, keys in (("glympse-patients", ["patient_id"]), ("glympse-meals", ["patient_id", "meal_id"])):
            resource.create_table(TableName=name, BillingMode="PAY_PER_REQUEST", KeySchema=[{"AttributeName": k, "KeyType": "HASH" if i == 0 else "RANGE"} for i,k in enumerate(keys)], AttributeDefinitions=[{"AttributeName": k, "AttributeType": "S"} for k in keys])
        self.storage = Accounts(resource)
        class Auth:
            def __init__(self): self.users = {}
            def signup(self, username, password):
                self.users[username] = password
                return {"user_id": username, "confirmed": True}
            def login(self, username, password):
                if self.users.get(username) != password: raise HTTPException(401)
                return {"access_token": username, "expires_in": 3600, "token_type": "bearer"}
            def authenticate(self, token):
                if token not in self.users: raise HTTPException(401)
                return {"sub": token, "username": token}
        self.auth = Auth()

    def make_client(self, directory, **options):
        options.setdefault("frontend_dir", "")
        options.setdefault("seed_recordings", [])
        options.setdefault("admin", ())
        return ApiClient(create_app(model_path=Path(__file__).parent / "artifacts/population.joblib", storage=self.storage, auth=self.auth, **options))

    def signed_up(self, client, username="alice"):
        client.post("/accounts", json={"username": username, "password": "test-password-123"})
        response = client.post("/sessions", json={"username": username, "password": "test-password-123"})
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    @staticmethod
    def seek_end(client, headers):
        """Move the replay clock to the end of the uploaded recording."""
        client.post("/live/control", headers=headers, json={"action": "seek", "minutes": 10 ** 7})

    def test_signup_upload_predict_daily_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, allow_signup=True)
            self.assertEqual(client.get("/health").json()["signup_open"], True)
            first = client.post("/accounts", json={"username": "alice", "password": "test-password-123"})
            self.assertEqual(first.status_code, 201)
            headers = {"Authorization": "Bearer " + client.post("/sessions", json={"username": "alice", "password": "test-password-123"}).json()["access_token"]}
            second = client.post("/accounts", json={"username": "bob", "password": "test-password-456"})
            other_headers = {"Authorization": "Bearer " + client.post("/sessions", json={"username": "bob", "password": "test-password-456"}).json()["access_token"]}
            self.assertEqual(client.get("/history", headers=other_headers).status_code, 409)
            self.assertEqual(client.get("/history").status_code, 401)
            raw = fixture(days=9)
            uploaded = client.post("/history", headers=headers,
                files={"file": ("history.csv", raw.to_csv(index=False), "text/csv")})
            self.seek_end(client, headers)
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            self.assertTrue(uploaded.json()["ready_for_day8"])
            request = {"meal_time": "2025-01-08T08:00:00Z", "dietary_intake": "Rice 100 g Egg 50 g", "confirm": True,
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

    def test_predict_takes_recent_cgm_from_uploaded_history(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, allow_signup=True)
            headers = self.signed_up(client)
            client.post("/history", headers=headers,
                        files={"file": ("history.csv", fixture(days=9).to_csv(index=False), "text/csv")})
            self.seek_end(client, headers)
            status = client.get("/history", headers=headers).json()
            self.assertEqual((status["first_reading"], status["last_reading"]), ("2025-01-01T00:00:00", "2025-01-09T23:55:00"))
            readings = client.get("/history/readings", headers=headers).json()
            self.assertEqual(len(readings["readings"]), 9 * 288)
            self.assertEqual(readings["events"][0], {"timestamp": "2025-01-01T08:00:00", "dietary_intake": "Rice 100 g Egg 50 g"})
            meal = {"meal_time": "2025-01-08T08:00:00", "dietary_intake": "Rice 100 g"}
            result = client.post("/predict", headers=headers, json=meal)
            self.assertEqual(result.status_code, 200, result.text)
            recent = result.json()["recent_cgm"]
            self.assertEqual((len(recent), recent[-1]["timestamp"]), (7, "2025-01-08T08:00:00"))
            explicit = client.post("/predict", headers=headers, json={**meal, "recent_cgm": recent}).json()
            self.assertEqual(explicit["predictions"], result.json()["predictions"])
            # Past the end of the replay there is nothing to start from: say why instead of guessing.
            missing = client.post("/predict", headers=headers, json={**meal, "meal_time": "2025-01-10T08:00:00"})
            self.assertEqual(missing.status_code, 422)
            self.assertIn("live clock is at Jan 09 23:55", missing.json()["detail"])
            other = self.signed_up(client, "bob")
            self.assertEqual(client.get("/history/readings", headers=other).status_code, 409)

    def test_legacy_admin_configuration_rejected_even_in_dev(self):
        for dev in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "Local admin passwords are retired"):
                    self.make_client(directory, admin=("synthetic-admin", "old-password"), dev=dev)

    def test_legacy_dataset_seeding_rejected_without_writes(self):
        for dev in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "account seeding is retired"):
                    self.make_client(directory, seed_recordings=["synthetic-recording"], dev=dev)
        with patch.dict(os.environ, {"GLYIMPSE_SEED_RECORDINGS": "synthetic-recording"}):
            with self.assertRaisesRegex(RuntimeError, "account seeding is retired"):
                create_app(storage=self.storage, auth=self.auth)
        self.assertEqual(self.storage.patients.scan()["Count"], 0)

    def test_spike_assessment_and_what_if_alternatives(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, allow_signup=True)
            headers = self.signed_up(client)
            client.post("/history", headers=headers,
                        files={"file": ("history.csv", fixture(days=9).to_csv(index=False), "text/csv")})
            self.seek_end(client, headers)
            meal = {"meal_time": "2025-01-08T08:00:00", "dietary_intake": "Rice 100 g, cake 80 g, orange juice 250 ml"}
            predicted = client.post("/predict", headers=headers, json=meal)
            self.assertEqual(predicted.status_code, 200, predicted.text)
            spike = predicted.json()["spike"]
            self.assertIn(spike["level"], {"none", "moderate", "high"})
            self.assertEqual(spike["sweet_items"], ["cake", "orange juice"])
            self.assertIn(spike["peak_minutes_after_meal"], {30, 60, 90, 120})
            self.assertTrue({d["feature"] for d in spike["food_drivers"]} <= {
                "food_starch", "food_sweet", "food_fruit", "food_amount_grams", "food_amount_ml", "sugar_adjustment"})
            # Cake 80 g (35% sugar) + orange juice 250 ml (8.4%) from the built-in table.
            sugar = predicted.json()["sugar_adjustment"]
            self.assertAlmostEqual(sugar["sugar_g"], 49.0, places=0)
            self.assertEqual({i["source"] for i in sugar["items"]}, {"table"})
            self.assertEqual(set(spike["thresholds"]), {"rise", "severe_rise", "personal", "meals"})
            self.assertIn("personal", client.get("/history", headers=headers).json())
            before = client.get("/explanations/daily", headers=headers).json()
            options = client.post("/predict/alternatives", headers=headers, json=meal)
            self.assertEqual(options.status_code, 200, options.text)
            body = options.json()
            labels = {row["label"]: row["dietary_intake"] for row in body["alternatives"]}
            self.assertEqual(labels["Skip the sweet items"], "Rice 100 g")
            self.assertEqual(labels["Half portion of sweet and starchy foods"], "Rice 50 g, cake 40 g, orange juice 125 ml")
            peaks = [row["spike"]["peak_glucose_mg_dl"] for row in body["alternatives"]]
            self.assertEqual(peaks, sorted(peaks))
            skipped = next(r for r in body["alternatives"] if r["label"] == "Skip the sweet items")
            self.assertLess(skipped["spike"]["peak_glucose_mg_dl"], spike["peak_glucose_mg_dl"])
            self.assertEqual(body["original"]["spike"]["peak_glucose_mg_dl"], spike["peak_glucose_mg_dl"])
            # What-if forecasts are not saved and don't change the daily ranking.
            self.assertEqual(client.get("/explanations/daily", headers=headers).json(), before)

    def test_live_replay_clock_hides_the_future(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, allow_signup=True)
            headers = self.signed_up(client)
            client.post("/history", headers=headers,
                        files={"file": ("history.csv", fixture(days=9).to_csv(index=False), "text/csv")})
            client.post("/live/control", headers=headers, json={"action": "pause"})
            state = client.post("/live/control", headers=headers, json={"action": "seek", "minutes": 8 * 60 + 2}).json()
            self.assertEqual((state["now"], state["day"], state["paused"]), ("2025-01-08T08:00:00", 8, True))
            self.assertEqual(state["readings"][-1]["timestamp"], "2025-01-08T08:00:00")
            self.assertEqual(len(state["readings"]), 7 * 288 + 97)
            newer = client.get("/live", headers=headers, params={"since": "2025-01-08T07:50:00"}).json()
            self.assertEqual([r["timestamp"] for r in newer["readings"]], ["2025-01-08T07:55:00", "2025-01-08T08:00:00"])
            self.assertEqual(client.get("/history/readings", headers=headers).json()["readings"][-1]["timestamp"],
                             "2025-01-08T08:00:00")
            # No meal time = now. A later meal time would peek at future readings.
            meal = {"dietary_intake": "Rice 100 g"}
            what_if = client.post("/predict", headers=headers, json=meal)
            self.assertEqual(what_if.status_code, 200, what_if.text)
            self.assertEqual(what_if.json()["meal_time"], "2025-01-08T08:00:00")
            self.assertEqual(client.get("/meals", headers=headers).json()["meals"], [])
            future = client.post("/predict", headers=headers, json={**meal, "meal_time": "2025-01-08T09:00:00"})
            self.assertEqual(future.status_code, 422)
            # Insulin and tablets from the dropdowns change the forecast inputs; confirm saves the meal.
            dosed = client.post("/predict", headers=headers, json={**meal, "confirm": True,
                "insulin": {"name": "Novolin R", "dose": 6}, "tablet": {"name": "Metformin", "dose": 500}}).json()
            self.assertEqual(dosed["medication"], {"Insulin dose - s.c.": "Novolin R, 6 IU",
                                                   "Non-insulin hypoglycemic agents": "metformin 500 mg"})
            saved = client.get("/meals", headers=headers).json()["meals"]
            self.assertEqual([(m["meal_time"], m["dietary_intake"]) for m in saved], [("2025-01-08T08:00:00", "Rice 100 g")])
            # The confirmed meal's own effect (with meal minus no meal) is added to the replay after it.
            effect = saved[0]["meal_effect"]
            self.assertEqual(len(effect), 4)
            client.post("/live/control", headers=headers, json={"action": "seek", "minutes": 9 * 60 + 2})
            after = {r["timestamp"]: r for r in client.get("/live", headers=headers).json()["readings"]}
            at_60 = after["2025-01-08T09:00:00"]
            self.assertAlmostEqual(at_60["glucose"] - at_60["recorded"], effect[1], places=0)
            self.assertNotIn("recorded", after["2025-01-08T08:00:00"])
            bad = client.post("/predict", headers=headers, json={**meal, "insulin": {"name": "Mystery", "dose": 6}})
            self.assertEqual(bad.status_code, 422)
            self.assertEqual(client.post("/live/control", headers=headers,
                                         json={"action": "speed", "seconds_per_reading": 7}).status_code, 422)
            restarted = client.post("/live/control", headers=headers, json={"action": "restart"}).json()
            self.assertEqual(restarted["now"], "2025-01-08T00:00:00")
            self.assertEqual(client.get("/meals", headers=headers).json()["meals"], [])
            catalog = client.get("/catalog").json()
            self.assertIn({"name": "Rice", "unit": "g", "portions": [50, 75, 80, 100, 150, 200]}, catalog["foods"])
            self.assertIn("Novolin R", [i["name"] for i in catalog["insulins"]])

    def test_meal_text_is_split_into_items(self):
        self.assertEqual(variants("Rice 100 g Egg 50 g")["variants"][0],
                         {"label": "Half portion of sweet and starchy foods", "dietary_intake": "Rice 50 g, Egg 50 g"})
        self.assertEqual(variants("cake")["variants"], [])
        self.assertEqual(variants("Noodles 200 g and honey tea")["sweet_items"], ["honey tea"])

    def test_model_accuracy_from_offline_results(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "day8.csv"
            pd.DataFrame({"Patient": [1, 1, 2, 2], "horizon_minutes": [30, 60, 30, 60],
                          "actual_glucose": [100., 150., 120., 200.], "predicted_glucose": [110., 140., 120., 230.],
                          "absolute_error_mg_dl": [10., 10., 0., 30.],
                          "within_20_mg_dl": [True, True, True, False],
                          "within_15_percent": [True, True, True, True]}).to_csv(results, index=False)
            client = self.make_client(directory, allow_signup=True, evaluation_path=str(results))
            headers = self.signed_up(client)
            self.assertEqual(client.get("/model/accuracy").status_code, 401)
            body = client.get("/model/accuracy", headers=headers).json()
            self.assertEqual((body["overall"]["forecasts"], body["overall"]["mae_mg_dl"]), (4, 12.5))
            self.assertEqual(body["overall"]["within_20_mg_dl_percent"], 75.0)
            self.assertEqual([h["horizon_minutes"] for h in body["horizons"]], [30, 60])
            self.assertEqual(len(body["points"]), 4)
            missing = self.make_client(directory, evaluation_path=str(Path(directory) / "none.csv"))
            self.assertEqual(missing.get("/model/accuracy", headers=headers).status_code, 404)

    def test_cors_allows_only_configured_origins(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, allowed_origins=["https://app.example"])
            def preflight(origin):
                return client.request("OPTIONS", "/history", headers={
                    "Origin": origin, "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "authorization"})
            self.assertEqual(preflight("https://app.example").headers.get("access-control-allow-origin"), "https://app.example")
            self.assertNotIn("access-control-allow-origin", preflight("https://evil.example").headers)

    def test_serves_frontend_next_to_api(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site"
            site.mkdir()
            (site / "index.html").write_text("<p>glympse</p>", encoding="utf-8")
            client = self.make_client(directory, frontend_dir=str(site))
            self.assertIn("glympse", client.get("/").text)
            self.assertEqual(client.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
