"""Tests for voice meal mapping (backend/food/meal.py + scripts/build_food_map.py meal mode).

No real network, microphone or model: ElevenLabs is an httpx.MockTransport, sounddevice and
Gemini are fakes, and USDA search/detail are patched.
"""
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

import httpx
from PIL import Image

from . import meal
from .meal import CandidateChoice, CandidateChoices, MealError, MealFood, MealInterpretation
from .usda import UsdaError

REPO = Path(__file__).resolve().parents[2]


def load_script():
    spec = importlib.util.spec_from_file_location("build_food_map", REPO / "scripts" / "build_food_map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def food(name, search_terms=None, quantity=None, unit=None, preparation=None, source="speech", warnings=()):
    return MealFood(name=name, preparation=preparation, quantity=quantity, unit=unit,
                    search_terms=search_terms or name, source=source, warnings=list(warnings))


class FakeGemini:
    """Returns queued responses per schema and records every call."""
    model = "fake-gemini"

    def __init__(self, interpretation=None, choices=None):
        self.responses = {MealInterpretation: interpretation, CandidateChoices: choices}
        self.calls = []

    def generate_json(self, prompt, schema, image=None):
        self.calls.append({"prompt": prompt, "schema": schema, "image": image})
        response = self.responses[schema]
        if isinstance(response, Exception):
            raise response
        return response


def jpeg_bytes(size=(64, 48)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (180, 90, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


# --- Transcription ---------------------------------------------------------------------

class TranscribeTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(meal, "RETRY_DELAY", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def client(self, *responses):
        self.requests = []
        queue = list(responses)

        def handler(request):
            self.requests.append(request)
            return queue.pop(0)
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_upload_uses_scribe_v2_auto_language_no_diarization_no_audio_tags(self):
        client = self.client(httpx.Response(200, json={"text": " two eggs and toast ", "language_code": "en"}))
        text = meal.transcribe(b"RIFFfake", "meal.wav", "audio/wav", "secret-key", client=client)
        self.assertEqual(text, "two eggs and toast")
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.elevenlabs.io/v1/speech-to-text")
        self.assertEqual(request.headers["xi-api-key"], "secret-key")
        body = request.content.decode("latin-1")
        for field, value in (("model_id", "scribe_v2"), ("diarize", "false"), ("tag_audio_events", "false")):
            self.assertRegex(body, rf'name="{field}"\r\n\r\n{value}\r\n')
        self.assertNotIn('name="language_code"', body)
        self.assertIn('name="file"; filename="meal.wav"', body)
        self.assertIn("RIFFfake", body)

    def test_auth_failure_names_the_variable(self):
        client = self.client(httpx.Response(401, json={"detail": "invalid"}))
        with self.assertRaisesRegex(MealError, "ELEVEN_LABS_API_KEY"):
            meal.transcribe(b"x", "a.wav", "audio/wav", "bad", client=client)

    def test_empty_transcript_is_an_error(self):
        client = self.client(httpx.Response(200, json={"text": "   "}))
        with self.assertRaisesRegex(MealError, "transcript is empty"):
            meal.transcribe(b"x", "a.wav", "audio/wav", "k", client=client)

    def test_transient_errors_are_retried_then_bounded(self):
        client = self.client(httpx.Response(503), httpx.Response(200, json={"text": "rice"}))
        self.assertEqual(meal.transcribe(b"x", "a.wav", "audio/wav", "k", client=client), "rice")
        client = self.client(*[httpx.Response(429)] * (meal.RETRIES + 1))
        with self.assertRaisesRegex(MealError, "HTTP 429"):
            meal.transcribe(b"x", "a.wav", "audio/wav", "k", client=client)
        self.assertEqual(len(self.requests), meal.RETRIES + 1)

    def test_key_is_read_from_exact_variable(self):
        with mock.patch.dict(os.environ, {"ELEVEN_LABS_API_KEY": " abc "}, clear=True):
            self.assertEqual(meal.elevenlabs_key(), "abc")
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "abc"}, clear=True):
            with self.assertRaisesRegex(MealError, "ELEVEN_LABS_API_KEY"):
                meal.elevenlabs_key()

    def test_gemini_key_and_model_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(MealError, "GEMINI_API_KEY"):
                meal.gemini_key()
            self.assertEqual(meal.gemini_model(), "gemini-3.8-flash")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g", "GEMINI_MODEL": "other"}, clear=True):
            self.assertEqual(meal.gemini_key(), "g")
            self.assertEqual(meal.gemini_model(), "other")


# --- Recording -------------------------------------------------------------------------

class FakeStream:
    def __init__(self, callback, fail_start=False):
        self.callback, self.fail_start = callback, fail_start
        self.started = self.closed = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("device busy")
        self.started = True
        self.callback(b"\x01\x00" * 800, 800, None, None)

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self, rate=16000.0, fail_start=False):
        self.rate, self.fail_start, self.streams, self.kwargs = rate, fail_start, [], None

    def query_devices(self, kind=None):
        return {"default_samplerate": self.rate}

    def RawInputStream(self, **kwargs):
        self.kwargs = kwargs
        stream = FakeStream(kwargs["callback"], self.fail_start)
        self.streams.append(stream)
        return stream


class InterruptingEvent:
    def wait(self, timeout):
        raise KeyboardInterrupt


class RecordTests(unittest.TestCase):
    def test_records_mono_wav_at_device_default_rate(self):
        sd = FakeSoundDevice(rate=44100.0)
        stop = threading.Event()
        stop.set()
        audio, seconds = meal.record_wav(stop, sd=sd)
        self.assertEqual(sd.kwargs["samplerate"], 44100)
        self.assertEqual(sd.kwargs["channels"], 1)
        self.assertTrue(sd.streams[0].closed)
        with wave.open(io.BytesIO(audio)) as handle:
            self.assertEqual((handle.getnchannels(), handle.getsampwidth(), handle.getframerate()), (1, 2, 44100))
            self.assertEqual(handle.getnframes(), 800)
        self.assertAlmostEqual(seconds, 800 / 44100)

    def test_ctrl_c_closes_the_microphone(self):
        sd = FakeSoundDevice()
        with self.assertRaises(KeyboardInterrupt):
            meal.record_wav(InterruptingEvent(), sd=sd)
        self.assertTrue(sd.streams[0].closed)

    def test_start_failure_closes_stream_and_explains(self):
        sd = FakeSoundDevice(fail_start=True)
        with self.assertRaisesRegex(MealError, "microphone"):
            meal.record_wav(threading.Event(), sd=sd)
        self.assertTrue(sd.streams[0].closed)

    def test_stops_at_max_duration(self):
        sd = FakeSoundDevice()
        audio, _ = meal.record_wav(threading.Event(), max_seconds=0, sd=sd)
        self.assertTrue(audio.startswith(b"RIFF"))
        self.assertTrue(sd.streams[0].closed)


# --- Files -----------------------------------------------------------------------------

class FileInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_image_is_validated_resized_and_reencoded(self):
        path = self.tmp / "big.png"
        Image.new("RGB", (4000, 2000), (10, 200, 10)).save(path)
        data = meal.prepare_image(path)
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(max(image.size), meal.MAX_IMAGE_SIDE)

    def test_invalid_or_missing_image_is_actionable(self):
        bad = self.tmp / "notes.jpg"
        bad.write_bytes(b"not an image")
        with self.assertRaisesRegex(MealError, "JPG, PNG or WebP"):
            meal.prepare_image(bad)
        gif = self.tmp / "anim.gif"
        Image.new("RGB", (8, 8)).save(gif)
        with self.assertRaisesRegex(MealError, "GIF"):
            meal.prepare_image(gif)
        with self.assertRaisesRegex(MealError, "not found"):
            meal.prepare_image(self.tmp / "missing.jpg")

    def test_audio_file_checks(self):
        with self.assertRaisesRegex(MealError, "not found"):
            meal.read_audio_file(self.tmp / "missing.wav")
        empty = self.tmp / "empty.wav"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(MealError, "empty"):
            meal.read_audio_file(empty)
        good = self.tmp / "meal.mp3"
        good.write_bytes(b"ID3data")
        self.assertEqual(meal.read_audio_file(good), (b"ID3data", "meal.mp3", "audio/mpeg"))


# --- Interpretation --------------------------------------------------------------------

class InterpretTests(unittest.TestCase):
    def test_voice_only_sends_no_image_and_forces_speech_source(self):
        gemini = FakeGemini(MealInterpretation(items=[food("toast", source="speech+photo")], warnings=[]))
        result = meal.interpret_meal("I had toast", gemini)
        self.assertIsNone(gemini.calls[0]["image"])
        self.assertIn("I had toast", gemini.calls[0]["prompt"])
        self.assertIn("No photo is attached", gemini.calls[0]["prompt"])
        self.assertEqual(result.items[0].source, "speech")

    def test_voice_plus_image_sends_photo_and_keeps_conflict_warnings(self):
        interpretation = MealInterpretation(
            items=[food("pasta", source="speech+photo", warnings=["Photo looks like rice, not pasta."])],
            warnings=["The photo also shows a salad that was not mentioned; ignored."])
        gemini = FakeGemini(interpretation)
        result = meal.interpret_meal("A bowl of pasta", gemini, image=b"jpeg")
        self.assertEqual(gemini.calls[0]["image"], b"jpeg")
        self.assertIn("Never add foods that appear only in the photo", gemini.calls[0]["prompt"])
        self.assertEqual(result.items[0].source, "speech+photo")
        self.assertEqual(len(result.items), 1)
        self.assertIn("salad", result.warnings[0])

    def test_prompt_carries_correction_and_exclusion_rules(self):
        gemini = FakeGemini(MealInterpretation(items=[food("eggs", quantity=3)], warnings=[]))
        transcript = "Two eggs, actually three, and fries but I didn't eat the fries"
        result = meal.interpret_meal(transcript, gemini)
        prompt = gemini.calls[0]["prompt"]
        self.assertIn("actually three", prompt)
        self.assertIn("I didn't eat the fries", prompt)
        self.assertIn("corrections and exclusions", prompt)
        self.assertEqual([(f.name, f.quantity) for f in result.items], [("eggs", 3)])

    def test_no_foods_is_an_error(self):
        gemini = FakeGemini(MealInterpretation(items=[], warnings=["Speaker talked about the weather."]))
        with self.assertRaisesRegex(MealError, "weather"):
            meal.interpret_meal("Nice day", gemini)

    def test_invalid_model_output_is_rejected(self):
        with self.assertRaisesRegex(MealError, "unexpected format"):
            meal.parse_model_output(MealInterpretation, '{"items": [{"name": "egg"}]}')
        with self.assertRaisesRegex(MealError, "unexpected format"):
            meal.parse_model_output(MealInterpretation, "not json")
        parsed = meal.parse_model_output(CandidateChoices, '{"choices": [{"item_index": 0, "fdc_id": null}]}')
        self.assertIsNone(parsed.choices[0].fdc_id)


# --- Matching --------------------------------------------------------------------------

CANDIDATES = [{"fdcId": 101, "description": "Egg, whole, fried", "dataType": "Survey (FNDDS)", "score": 9},
              {"fdcId": 102, "description": "Egg, whole, raw", "dataType": "SR Legacy", "score": 8}]


class MatchTests(unittest.TestCase):
    def test_food_map_exact_match_skips_usda(self):
        food_map = {"french_fries": {"fdcId": 1, "description": "French fries", "dataType": "Survey (FNDDS)"}}

        def search(term):
            raise AssertionError("USDA searched despite a food_map hit")
        items = meal.match_items([food("French fries", "french fries")], food_map, search, FakeGemini())
        self.assertEqual(items[0]["match_status"], "mapped")
        self.assertEqual(items[0]["match"], {"fdcId": 1, "description": "French fries", "dataType": "Survey (FNDDS)"})

    def test_live_candidates_choice_is_validated(self):
        gemini = FakeGemini(choices=CandidateChoices(choices=[
            CandidateChoice(item_index=0, fdc_id=101), CandidateChoice(item_index=1, fdc_id=999)]))
        foods = [food("fried egg", "egg fried", preparation="fried"), food("kale chips", "kale chips")]
        items = meal.match_items(foods, {}, lambda term: CANDIDATES, gemini)
        self.assertEqual(items[0]["match_status"], "matched")
        self.assertEqual(items[0]["match"]["fdcId"], 101)
        self.assertEqual(items[1]["match_status"], "unmatched")
        self.assertIsNone(items[1]["match"])
        self.assertTrue(any("outside" in w for w in items[1]["warnings"]))
        listing = gemini.calls[0]["prompt"]
        self.assertIn('"fdc_id": 101', listing)
        self.assertIn('"preparation": "fried"', listing)

    def test_unmatched_and_failed_searches_are_preserved_in_order(self):
        def search(term):
            if term == "dragon fruit":
                return []
            raise UsdaError("HTTP 500")
        foods = [food("dragon fruit"), food("mystery stew", "stew")]
        gemini = FakeGemini()
        items = meal.match_items(foods, {}, search, gemini)
        self.assertEqual([i["name"] for i in items], ["dragon fruit", "mystery stew"])
        self.assertEqual([i["match_status"] for i in items], ["unmatched", "error"])
        self.assertEqual(items[1]["search_terms"], "stew")
        self.assertIn("USDA search failed", items[1]["warnings"][0])
        self.assertEqual(gemini.calls, [])

    def test_null_choice_and_selection_failure_leave_items_unmatched(self):
        gemini = FakeGemini(choices=CandidateChoices(choices=[CandidateChoice(item_index=0, fdc_id=None)]))
        items = meal.match_items([food("egg")], {}, lambda term: CANDIDATES, gemini)
        self.assertEqual(items[0]["match_status"], "unmatched")
        gemini = FakeGemini(choices=MealError("Gemini request failed (HTTP 500)"))
        items = meal.match_items([food("egg")], {}, lambda term: CANDIDATES, gemini)
        self.assertEqual(items[0]["match_status"], "unmatched")
        self.assertIn("Could not choose", items[0]["warnings"][0])


# --- Nutrition -------------------------------------------------------------------------

RECORD = {"fdcId": 101, "description": "Rice", "source": "SR Legacy",
          "per_100g": {"kcal": 130.0, "protein_g": 2.7, "fat_g": 0.3, "carbs_g": 28.0, "fiber_g": None, "sugar_g": None},
          "portions": [{"label": "1 cup", "grams": 158.0}, {"label": "1/2 cup", "grams": 79.0}],
          "default_portion": {"label": "1 cup", "grams": 158.0}, "missing": [], "kcal_estimated": False}


def matched_item(quantity, unit):
    return {"name": "rice", "quantity": quantity, "unit": unit, "warnings": [],
            "match": {"fdcId": 101, "description": "Rice", "dataType": "SR Legacy"}}


class NutritionTests(unittest.TestCase):
    def test_counted_eggs_scale_selected_usda_reference(self):
        record = {**RECORD, "portions": [{"label": "1 cup", "grams": 200}, {"label": "1 large", "grams": 50}]}
        gemini = mock.Mock()
        gemini.generate_json.return_value = meal.PortionChoices(choices=[
            meal.PortionChoice(item_index=0, portion_index=1)])
        for unit in ("egg", None):
            item = {**matched_item(3, unit), "name": "eggs"}
            meal.attach_nutrition([item], {"101": record}, gemini=gemini)
            serving = item["nutrition"]["serving"]
            self.assertEqual(serving["grams"], 150)
            self.assertEqual(serving["portion_count"], 3)
            self.assertEqual(serving["portion_index"], 1)
            self.assertEqual(serving["nutrients"]["kcal"], 195)
            self.assertEqual(item["quantity"], 3)

    def test_fraction_of_multi_item_reference_and_multiple_foods(self):
        record = {**RECORD, "portions": [{"label": "2 pieces", "grams": 40}]}
        items = [matched_item(5, "pieces"), matched_item(3, "pieces")]
        meal.attach_nutrition(items, {"101": record})
        self.assertEqual([i["nutrition"]["serving"]["grams"] for i in items], [100, 60])
        self.assertEqual(items[0]["nutrition"]["serving"]["portion_count"], 2.5)

    def test_slice_is_not_matched_to_whole_pizza_description(self):
        portions = [{"label": "1 pizza (8 slices)", "grams": 800}, {"label": "1 slice", "grams": 100}]
        self.assertEqual(meal.serving_grams(2, "slices", portions)[0], 200)

    def test_plain_slice_preferred_over_partially_eaten_reference(self):
        portions = [{"label": "1 slice, crust not eaten", "grams": 12}, {"label": "1 slice", "grams": 28}]
        self.assertEqual(meal.serving_grams(2, "slices", portions)[0], 56)

    def test_invalid_or_failed_portion_choice_preserves_count_without_guessing(self):
        record = {**RECORD, "portions": [{"label": "1 large", "grams": 50}]}
        gemini = mock.Mock()
        for index in (99, None):
            gemini.generate_json.return_value = meal.PortionChoices(choices=[
                meal.PortionChoice(item_index=0, portion_index=index)])
            item = {**matched_item(3, None), "name": "eggs"}
            meal.attach_nutrition([item], {"101": record}, gemini=gemini)
            self.assertIsNone(item["nutrition"]["serving"])
            self.assertEqual(item["quantity"], 3)
        gemini.generate_json.side_effect = meal.MealError("Unavailable")
        meal.attach_nutrition([item], {"101": record}, gemini=gemini)
        self.assertIsNone(item["nutrition"]["serving"])
        self.assertIn("Could not resolve", item["warnings"][0])

    def test_stated_weight_scales_per_100g(self):
        item = meal.attach_nutrition([matched_item(200, "grams")], {"101": RECORD})[0]
        self.assertEqual(item["nutrition"]["per_100g"], RECORD["per_100g"])
        self.assertEqual(item["nutrition"]["serving"]["grams"], 200)
        self.assertEqual(item["nutrition"]["serving"]["basis"], "stated weight")
        self.assertEqual(item["nutrition"]["serving"]["nutrients"]["kcal"], 260)

    def test_compatible_usda_portion_converts(self):
        item = meal.attach_nutrition([matched_item(2, "cups")], {"101": RECORD})[0]
        self.assertEqual(item["nutrition"]["serving"]["grams"], 316.0)
        self.assertIn("1 cup", item["nutrition"]["serving"]["basis"])

    def test_unknown_amount_or_unit_stays_unknown(self):
        items = meal.attach_nutrition([matched_item(None, None), matched_item(2, "bowl")], {"101": RECORD})
        for item in items:
            self.assertIsNone(item["nutrition"]["serving"])
            self.assertEqual(item["nutrition"]["per_100g"]["kcal"], 130.0)

    def test_missing_record_and_unmatched_items(self):
        unmatched = {"name": "x", "quantity": None, "unit": None, "warnings": [], "match": None}
        items = meal.attach_nutrition([matched_item(1, "cup"), unmatched], {})
        self.assertIsNone(items[0]["nutrition"])
        self.assertIn("Nutrition is unavailable", items[0]["warnings"][0])
        self.assertIsNone(items[1]["nutrition"])

    def test_fractional_portion_labels(self):
        portions = [{"label": "1 1/2 cup, chopped", "grams": 150.0}]
        self.assertEqual(meal.serving_grams(1, "cup", portions)[0], 100.0)


# --- CLI -------------------------------------------------------------------------------

class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "data"
        self.data.mkdir()
        (self.data / "food_map.json").write_text(json.dumps(
            {"toast": {"fdcId": 7, "description": "Bread, toasted", "dataType": "SR Legacy"}}), encoding="utf-8")
        (self.data / "food_map_review.csv").write_text("label,display_name\n", encoding="utf-8")
        (self.data / "nutrient_cache.json").write_text(json.dumps({"7": dict(RECORD, fdcId=7)}), encoding="utf-8")
        self.audio = self.tmp / "meal.wav"
        self.audio.write_bytes(meal.encode_wav(b"\x00\x00" * 100, 16000))
        patches = [mock.patch.object(self.script, "DATA_DIR", self.data),
                   mock.patch.dict(os.environ, {"ELEVEN_LABS_API_KEY": "el", "GEMINI_API_KEY": "gm"})]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def snapshot(self):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.data.iterdir()}

    def run_main(self, *argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            self.script.main(list(argv))

    def assert_usage_error(self, *argv):
        with self.assertRaises(SystemExit) as raised:
            self.run_main(*argv)
        self.assertEqual(raised.exception.code, 2)

    def test_invalid_meal_combinations(self):
        self.assert_usage_error("--record", "--labels", "pizza")
        self.assert_usage_error("--audio", "x.wav", "--out", str(self.tmp))
        self.assert_usage_error("--record", "--audio", "x.wav")
        self.assert_usage_error("--image", "meal.jpg")
        self.assert_usage_error("--meal-out", "meal.json")

    def fake_pipeline(self, gemini, transcript="I had toast and a fried egg"):
        return [mock.patch.object(meal, "Gemini", lambda key: gemini),
                mock.patch.object(meal, "transcribe", lambda *a, **k: transcript),
                mock.patch.object(self.script, "candidates", lambda term, client: CANDIDATES),
                mock.patch.object(self.script, "fetch_nutrients",
                                  lambda food_map, client, pause: {"101": dict(RECORD, fdcId=101)})]

    def test_audio_and_image_meal_writes_only_the_meal_file(self):
        image = self.tmp / "meal.jpg"
        image.write_bytes(jpeg_bytes())
        gemini = FakeGemini(
            MealInterpretation(items=[food("toast", quantity=1, unit="cup"),
                                      food("fried egg", "egg fried", preparation="fried", source="speech+photo")],
                               warnings=[]),
            CandidateChoices(choices=[CandidateChoice(item_index=1, fdc_id=101)]))
        before = self.snapshot()
        out = self.tmp / "out" / "meal.json"
        patches = self.fake_pipeline(gemini)
        for patcher in patches:
            patcher.start()
        try:
            self.run_main("--audio", str(self.audio), "--image", str(image), "--fetch", "--meal-out", str(out))
        finally:
            for patcher in patches:
                patcher.stop()
        self.assertEqual(self.snapshot(), before)
        document = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 1)
        self.assertTrue(document["image_used"])
        self.assertEqual(document["transcript"], "I had toast and a fried egg")
        self.assertEqual([i["match_status"] for i in document["items"]], ["mapped", "matched"])
        self.assertEqual(document["items"][0]["nutrition"]["serving"]["grams"], 158.0)
        self.assertEqual(document["items"][1]["match"]["fdcId"], 101)
        self.assertIsNone(document["items"][1]["nutrition"]["serving"])
        self.assertIsNotNone(gemini.calls[0]["image"])
        self.assertEqual([p.name for p in out.parent.iterdir()], ["meal.json"])  # no temp files left

    def test_interpretation_failure_writes_nothing(self):
        gemini = FakeGemini(MealError("Gemini returned output in an unexpected format (items: x)."))
        out = self.tmp / "out" / "meal.json"
        patches = self.fake_pipeline(gemini)
        for patcher in patches:
            patcher.start()
        try:
            with self.assertRaises(SystemExit) as raised:
                self.run_main("--audio", str(self.audio), "--meal-out", str(out))
        finally:
            for patcher in patches:
                patcher.stop()
        self.assertEqual(raised.exception.code, 1)
        self.assertFalse(out.exists())

    def test_meal_out_cannot_target_shared_files(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main("--audio", str(self.audio), "--meal-out", str(self.data / "food_map.json"))
        self.assertEqual(raised.exception.code, 1)

    def test_label_mapping_mode_is_unchanged(self):
        found = [{"fdcId": 5, "description": "Pizza, cheese", "dataType": "Survey (FNDDS)", "score": 3.0}]
        out = self.tmp / "labels"
        with mock.patch.object(self.script, "candidates", lambda label, client: found):
            self.run_main("--labels", "pizza", "--out", str(out))
        self.assertEqual(json.loads((out / "food_map.json").read_text(encoding="utf-8")),
                         {"pizza": {"fdcId": 5, "description": "Pizza, cheese", "dataType": "Survey (FNDDS)"}})
        header = (out / "food_map_review.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "label,display_name,rank,fdcId,description,dataType,score")


if __name__ == "__main__":
    unittest.main()
