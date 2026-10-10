import json
from pathlib import Path

import openai
import pytest

from src.cache import JsonCache, ProgressJournal, RunLock, RunLockError
from src.config import GEMINI_OPENAI_BASE_URL, PROVIDER_GEMINI, Settings
from src.llm_client import LLMClient, UsageTracker, _retry_after_seconds
from src.parse_meals import MealParser

from .conftest import FakeOpenAI
from .test_parser import MEALS, good_handler


def test_run_lock_blocks_concurrent_runs(tmp_path):
    lock_path = tmp_path / ".run.lock"
    with RunLock(lock_path):
        with pytest.raises(RunLockError):
            with RunLock(lock_path):
                pass
    with RunLock(lock_path):
        pass


def test_atomic_write_retries_when_temp_file_vanishes(tmp_path, monkeypatch):
    import src.cache as cache_mod

    real_replace = cache_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            Path(src).unlink()
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return real_replace(src, dst)

    monkeypatch.setattr(cache_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(cache_mod.time, "sleep", lambda s: None)
    target = tmp_path / "c.json"
    cache_mod.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert calls["n"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_checkpoint_does_not_abort_and_retries(tmp_path, monkeypatch):
    import src.cache as cache_mod

    cache = JsonCache(tmp_path / "c.json", autosave_every=1)

    def broken(path, data):
        raise PermissionError("locked")

    monkeypatch.setattr(cache_mod, "atomic_write_json", broken)
    cache.set("a", 1)
    monkeypatch.undo()
    cache.set("b", 2)
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_periodic_snapshots_respect_min_interval(tmp_path):
    cache = JsonCache(tmp_path / "c.json", snapshot_dir=tmp_path / "snaps", snapshot_every=1,
                      snapshot_min_interval_s=3600)
    for i in range(10):
        cache.set(f"k{i}", i)
    assert len(cache.snapshots()) == 1
    cache.snapshot("stage")
    assert len(cache.snapshots()) == 2


def test_snapshots_rotate_and_keep_newest(tmp_path):
    snaps = tmp_path / "snaps"
    cache = JsonCache(tmp_path / "c.json", snapshot_dir=snaps, snapshot_every=2, snapshot_keep=3)
    for i in range(10):
        cache.set(f"k{i}", i)  # autosave_every=1 -> one flush per set
    files = cache.snapshots()
    assert len(files) == 3
    assert json.loads(files[-1].read_text(encoding="utf-8")) == {f"k{i}": i for i in range(10)}


def test_corrupt_cache_restores_latest_snapshot(tmp_path):
    snaps = tmp_path / "snaps"
    cache = JsonCache(tmp_path / "c.json", snapshot_dir=snaps)
    cache.set("a", 1)
    cache.snapshot()
    cache.set("b", 2)
    (tmp_path / "c.json").write_text("{truncated", encoding="utf-8")

    restored = JsonCache(tmp_path / "c.json", snapshot_dir=snaps)
    assert restored.data == {"a": 1}
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8")) == {"a": 1}
    assert list(tmp_path.glob("c.json.corrupt-*"))


def test_journal_appends_events(tmp_path):
    journal = ProgressJournal(tmp_path / "progress.jsonl", run_id="r1")
    journal.log("start", meals=3)
    journal.log("done")
    lines = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines] == ["start", "done"]
    assert lines[0]["run_id"] == "r1" and lines[0]["meals"] == 3


def test_parser_stops_after_consecutive_failures_and_keeps_completed_work(settings):
    settings.llm_max_consecutive_failures = 2
    texts = {str(i): f"Rice {100 + i} g" for i in range(5)}

    def handler(payload, response_format):
        if payload[0]["meal_text"] == texts["0"]:
            return good_handler([{"meal_id": payload[0]["meal_id"], "meal_text": MEALS["1"]}], response_format)
        return openai.RateLimitError("quota exhausted", response=_FakeResponse(), body=None)

    fake = FakeOpenAI(handler)
    llm = LLMClient(settings, UsageTracker(None, None), client=fake, sleep=lambda s: None)
    parser = MealParser(JsonCache(settings.parsed_cache_path), settings, llm=llm, batch_size=1)
    out = parser.parse_meals(list(texts.items()))
    # meal 0 is parsed (batch + single retry), meals 1-2 hit the quota, then the breaker stops further calls
    assert "0" in out.parsed
    assert len(fake.calls) == 2 + 2 * settings.max_retries
    assert out.failures["1"].startswith("batch request failed")
    assert out.failures["3"] == out.failures["4"] == "skipped: LLM unavailable"
    assert JsonCache(settings.parsed_cache_path).get(parser._cache_key(texts["0"])) is not None


class _FakeResponse:
    status_code = 429
    headers = {}
    request = None


def test_gemini_retry_delay_is_parsed():
    exc = Exception('429 RESOURCE_EXHAUSTED ... "retryDelay": "37s"')
    assert _retry_after_seconds(exc) == 37.0


def test_settings_pick_gemini_when_only_google_key(monkeypatch):
    for var in ("OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "GEMINI_MODEL", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_AI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_API_AI", "g-key")
    monkeypatch.setattr("src.config.load_env", lambda: None)
    s = Settings.from_env()
    assert s.llm_provider == PROVIDER_GEMINI
    assert s.llm_api_key == "g-key" and s.llm_base_url == GEMINI_OPENAI_BASE_URL
    assert s.llm_model == "gemini-2.5-flash" and s.llm_requests_per_minute == 8.0
