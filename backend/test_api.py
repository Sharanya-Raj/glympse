import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx
import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

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
        cohort = pd.concat([fixture("other1", 7), fixture("other2", 7)], ignore_index=True)
        examples, features, _ = meal_examples(prepared_history(cohort))
        model, _ = fit_population(examples, features, None)
        cls.bundle = {"model": model, "features": features, "schema": 1, "version": "test"}

    def make_client(self, directory, **options):
        path = Path(directory)
        artifact = path / "population.joblib"
        joblib.dump(self.bundle, artifact)
        options.setdefault("frontend_dir", "")
        options.setdefault("seed_recordings", [])
        options.setdefault("admin", ())
        return ApiClient(create_app(path, artifact, **options))

    def signed_up(self, client, username="alice"):
        response = client.post("/accounts", json={"username": username, "password": "test-password-123"})
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
            headers = {"Authorization": "Bearer " + first.json()["access_token"]}
            second = client.post("/accounts", json={"username": "bob", "password": "test-password-456"})
            other_headers = {"Authorization": "Bearer " + second.json()["access_token"]}
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

    def test_admin_only_login_signout_and_rekey(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, admin=("Glympse", "admin-password-123"), allow_signup=False)
            blocked = client.post("/accounts", json={"username": "mallory", "password": "test-password-123"})
            self.assertEqual(blocked.status_code, 403)
            self.assertEqual(client.post("/sessions", json={"username": "Glympse", "password": "wrong-password-1"}).status_code, 401)
            login = client.post("/sessions", json={"username": "Glympse", "password": "admin-password-123"})
            headers = {"Authorization": "Bearer " + login.json()["access_token"]}
            self.assertEqual(client.get("/me", headers=headers).json(), {"username": "glympse", "admin": True})
            self.assertEqual(client.request("DELETE", "/sessions", headers=headers).status_code, 204)
            self.assertEqual(client.get("/me", headers=headers).status_code, 401)
            # Restarting with a new configured password re-keys the account and ends its sessions.
            login = client.post("/sessions", json={"username": "Glympse", "password": "admin-password-123"})
            old = {"Authorization": "Bearer " + login.json()["access_token"]}
            restarted = self.make_client(directory, admin=("Glympse", "new-admin-password-1"))
            self.assertEqual(restarted.get("/me", headers=old).status_code, 401)
            self.assertEqual(restarted.post("/sessions", json={"username": "glympse", "password": "admin-password-123"}).status_code, 401)
            self.assertEqual(restarted.post("/sessions", json={"username": "glympse", "password": "new-admin-password-1"}).status_code, 200)

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

    def test_dev_admin_may_use_short_password_only_in_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.make_client(directory, admin=("admin@12345", "dev123"), dev=True)
            login = client.post("/sessions", json={"username": "admin@12345", "password": "dev123"})
            self.assertEqual(login.status_code, 200, login.text)
            self.assertEqual(client.post("/sessions", json={"username": "admin@12345", "password": "dev124"}).status_code, 401)
            with self.assertRaises(ValidationError):
                self.make_client(directory, admin=("admin@12345", "dev123"), dev=False)

    def test_dev_seeds_patient_accounts_from_dataset_workbooks(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            workbook = fixture(days=8)
            workbook["Date"] = workbook.Date.dt.strftime("%Y-%m-%d %H:%M:%S")
            workbook.drop(columns=["Patient", "Recording"]).to_excel(data / "9001_0_20250101.xlsx", index=False)
            workbook.drop(columns=["Patient", "Recording"]).to_excel(data / "9002_0_20250101.xlsx", index=False)
            seeded = dict(seed_recordings=["9001_0_20250101", "9002_0_20250101:own-pass"],
                          seed_password="dev123", data_dir=str(data))
            client = self.make_client(directory, dev=True, **seeded)
            own = client.post("/sessions", json={"username": "patient9002", "password": "own-pass"})
            self.assertEqual(own.status_code, 200, own.text)
            self.assertEqual(client.post("/sessions", json={"username": "patient9002", "password": "dev123"}).status_code, 401)
            login = client.post("/sessions", json={"username": "patient9001", "password": "dev123"})
            self.assertEqual(login.status_code, 200, login.text)
            headers = {"Authorization": "Bearer " + login.json()["access_token"]}
            status = client.get("/history", headers=headers).json()
            self.assertEqual((status["rows"], status["ready_for_day8"], status["revision"]), (8 * 288, True, 1))
            self.assertEqual(client.get("/me", headers=headers).json(), {"username": "patient9001", "admin": False})
            # Restarting doesn't reload or overwrite an account that already has history.
            again = self.make_client(directory, dev=True, **seeded)
            self.assertEqual(again.get("/history", headers=headers).json()["revision"], 1)
            # A username in the config renames the existing account; its history comes along.
            renamed = self.make_client(directory, dev=True, **{**seeded, "seed_recordings": ["9001_0_20250101=jane"]})
            self.assertEqual(renamed.get("/me", headers=headers).json()["username"], "jane")
            self.assertEqual(renamed.get("/history", headers=headers).json()["revision"], 1)
            self.assertEqual(renamed.post("/sessions", json={"username": "patient9001", "password": "dev123"}).status_code, 401)
            with self.assertRaises(RuntimeError):
                self.make_client(directory, dev=False, **seeded)

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
