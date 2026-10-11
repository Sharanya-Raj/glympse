"""Dictation API integration with fake provider responses; no paid/network calls."""
import os
import unittest
from unittest.mock import Mock, patch

from fastapi import Depends, FastAPI
from fastapi.security import HTTPBearer

from . import meal, routes
from .cache import FoodCache
from .test_food import ApiClient
from .test_meal import RECORD, food


class DictationTests(unittest.TestCase):
    def setUp(self):
        self.cache = FoodCache(data_dir="/nonexistent")
        self.cache.food_map = {"rice": {"fdcId": 101, "description": "Rice", "dataType": "SR Legacy"}}
        self.cache.nutrients = {"101": RECORD}
        self.gemini = Mock(model="test-gemini")
        self.gemini.generate_json.side_effect = [
            meal.MealInterpretation(items=[food("rice", quantity=2, unit="cups"), food("egg")], warnings=[]),
            meal.CandidateChoices(choices=[meal.CandidateChoice(item_index=1, fdc_id=102)])]
        self.transcribe = Mock(return_value="I ate two cups of rice and an egg")
        patches = [patch.dict(os.environ, {"ELEVEN_LABS_API_KEY": "test-el", "GEMINI_API_KEY": "test-gm",
                                          "USDA_API_KEY": "test-usda"}),
                   patch.object(meal, "Gemini", return_value=self.gemini),
                   patch.object(meal, "transcribe", self.transcribe),
                   patch.object(routes.usda, "search_foods", return_value=[
                       {"fdcId": 102, "description": "Egg, cooked", "dataType": "SR Legacy"}]),
                   patch.object(routes.usda, "food_detail", return_value={
                       "fdcId": 102, "description": "Egg, cooked", "foodNutrients": [], "dataType": "SR Legacy"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        app = FastAPI()
        bearer = HTTPBearer()

        def authenticated(credentials=Depends(bearer)):
            return "test-user"

        app.include_router(routes.build_food_router(authenticated, cache=self.cache))
        self.client = ApiClient(app)

    def request(self, content=b"test audio", mime="audio/webm;codecs=opus", auth=True):
        return self.client.request("POST", "/food/dictate", files={"audio": ("meal.webm", content, mime)},
                                   headers={"Authorization": "Bearer test"} if auth else {})

    def test_transcript_to_reviewable_foods_with_usda_nutrition(self):
        response = self.request()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["transcript"], self.transcribe.return_value)
        self.assertEqual([i["match_status"] for i in body["items"]], ["mapped", "matched"])
        self.assertEqual(body["items"][0]["nutrition"]["serving"]["grams"], 316)
        self.assertEqual(body["items"][1]["nutrition"]["serving"]["grams"], 100)
        self.assertTrue(body["items"][1]["nutrition"]["serving"]["assumed"])
        self.assertEqual(body["items"][1]["match"]["fdcId"], 102)
        self.transcribe.assert_called_once_with(b"test audio", "meal.webm", "audio/webm", "test-el")
        self.gemini.close.assert_called_once()
        self.assertNotIn("test-el", response.text)

    def test_auth_and_upload_validation_precede_provider_calls(self):
        self.assertIn(self.request(auth=False).status_code, [401, 403])
        self.assertEqual(self.request(mime="text/plain").status_code, 415)
        self.assertEqual(self.request(content=b"").status_code, 422)
        with patch.object(routes, "MAX_DICTATION_BYTES", 5):
            self.assertEqual(self.request().status_code, 413)
        self.transcribe.assert_not_called()

    def test_counted_foods_reach_response_with_full_quantity(self):
        self.cache.food_map = {"eggs": {"fdcId": 101, "description": "Egg, cooked", "dataType": "SR Legacy"}}
        self.cache.nutrients["101"] = {**RECORD, "portions": [{"label": "1 large", "grams": 50}]}
        self.gemini.generate_json.side_effect = [
            meal.MealInterpretation(items=[food("eggs", quantity=3, unit=None)], warnings=[]),
            meal.PortionChoices(choices=[meal.PortionChoice(item_index=0, portion_index=0)])]
        response = self.request()
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["items"][0]
        self.assertEqual(item["quantity"], 3)
        self.assertEqual(item["nutrition"]["serving"]["grams"], 150)
        self.assertEqual(item["nutrition"]["serving"]["portion_count"], 3)

    def test_missing_configuration_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.request()
        self.assertEqual(response.status_code, 503)
        self.assertIn("ELEVEN_LABS_API_KEY", response.json()["detail"])
        self.transcribe.assert_not_called()

    def test_transcription_failure_closes_client(self):
        self.transcribe.side_effect = meal.MealError("The transcript is empty; try again.")
        response = self.request()
        self.assertEqual(response.status_code, 502)
        self.assertIn("transcript is empty", response.json()["detail"])
        self.gemini.close.assert_called_once()

    def test_usda_outage_keeps_transcript_and_unmatched_foods(self):
        with patch.object(routes.usda, "search_foods", side_effect=routes.usda.UsdaError("Unavailable")):
            response = self.request()
        self.assertEqual(response.status_code, 200, response.text)
        egg = response.json()["items"][1]
        self.assertEqual(egg["match_status"], "error")
        self.assertIsNone(egg["nutrition"])
        self.assertIn("USDA search failed", egg["warnings"][0])


if __name__ == "__main__":
    unittest.main()
