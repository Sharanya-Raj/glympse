"""Thin LLM wrapper: structured-output parsing, throttling, retries with backoff, token/cost tracking.

Works with OpenAI and with Gemini through Google's OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings
from .models import LLMUsage
from .retry import with_retries

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class LLMResponseError(Exception):
    """The model answered, but the answer was unusable (refusal, truncated, unparseable)."""


class LLMUnavailableError(Exception):
    """Too many consecutive batches failed (e.g. quota exhausted); stop calling the LLM for now."""


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    return input_tokens / 1e6 * input_cost_per_million + output_tokens / 1e6 * output_cost_per_million


class UsageTracker:
    def __init__(self, input_cost_per_million: float | None, output_cost_per_million: float | None) -> None:
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.usage = LLMUsage()

    def record(self, purpose: str, usage: Any) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        self.usage.requests += 1
        self.usage.input_tokens += prompt
        self.usage.output_tokens += completion
        bucket = self.usage.by_purpose.setdefault(
            purpose, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        )
        bucket["requests"] += 1
        bucket["input_tokens"] += prompt
        bucket["output_tokens"] += completion

    def record_failure(self) -> None:
        self.usage.failed_requests += 1

    @property
    def estimated_cost(self) -> float | None:
        return estimate_cost(
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.input_cost_per_million,
            self.output_cost_per_million,
        )

    def snapshot(self) -> LLMUsage:
        return self.usage.model_copy(update={"estimated_cost_usd": self.estimated_cost})


class FailureBreaker:
    """Counts consecutive failed batches and raises LLMUnavailableError past a limit."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.consecutive = 0

    def success(self) -> None:
        self.consecutive = 0

    def failure(self, what: str) -> None:
        self.consecutive += 1
        if self.consecutive >= self.limit:
            raise LLMUnavailableError(
                f"{self.consecutive} consecutive {what} batches failed; stopping LLM calls for this run "
                "(completed work is saved; rerun later to resume)"
            )


def _retryable_errors() -> tuple[type[BaseException], ...]:
    import openai

    return (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        LLMResponseError,
        ValidationError,
        json.JSONDecodeError,
    )


_RETRY_DELAY_RE = re.compile(r"retry(?:Delay|[ _-]in)[\"':\s]*([\d.]+)\s*s", re.IGNORECASE)


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after")
        try:
            if value is not None:
                return float(value)
        except ValueError:
            pass
    # Gemini puts the hint in the error body, e.g. "retryDelay": "37s"
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) if match else None


class LLMClient:
    """Calls a chat model and returns a validated Pydantic object."""

    def __init__(
        self,
        settings: Settings,
        usage: UsageTracker,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.usage = usage
        self.sleep = sleep
        if client is None:
            if not settings.llm_api_key:
                raise RuntimeError(f"No API key configured for LLM provider {settings.llm_provider!r}")
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                max_retries=0,  # retries are handled here so every attempt is logged
                timeout=settings.llm_timeout_s,
            )
        self.client = client
        self.model = settings.llm_model
        rpm = settings.llm_requests_per_minute
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._last_request = 0.0

    def _parse_fn(self):
        completions = self.client.chat.completions
        if hasattr(completions, "parse"):
            return completions.parse
        return self.client.beta.chat.completions.parse

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self._last_request + self.min_interval - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last_request = time.monotonic()

    def parse(self, *, system: str, user: str, response_format: type[M], purpose: str) -> M:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": response_format,
        }
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature
        if self.settings.llm_reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort

        def call() -> M:
            self._throttle()
            try:
                completion = self._parse_fn()(**kwargs)
            except Exception as exc:
                self.usage.record_failure()
                if type(exc).__name__ in ("LengthFinishReasonError", "ContentFilterFinishReasonError"):
                    raise LLMResponseError(str(exc)) from exc
                raise
            self.usage.record(purpose, getattr(completion, "usage", None))
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise LLMResponseError(f"model refused: {message.refusal}")
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                raise LLMResponseError("response contained no parsed content")
            if not isinstance(parsed, response_format):
                parsed = response_format.model_validate(parsed)
            return parsed

        return with_retries(
            call,
            retry_on=_retryable_errors(),
            max_attempts=self.settings.max_retries,
            base_s=self.settings.backoff_base_s,
            max_s=self.settings.backoff_max_s,
            description=f"LLM {purpose} request",
            retry_after=_retry_after_seconds,
            sleep=self.sleep,
        )
