"""Exponential backoff with jitter for network calls."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def backoff_delay(attempt: int, base_s: float, max_s: float) -> float:
    """Delay before retry number `attempt` (1-based), with +/-50% jitter."""
    return min(max_s, base_s * 2 ** (attempt - 1)) * (0.5 + random.random())


def with_retries(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    max_attempts: int,
    base_s: float,
    max_s: float,
    description: str,
    retry_after: Callable[[BaseException], float | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying on `retry_on` exceptions. Re-raises the last exception when attempts run out."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == max_attempts:
                logger.error("%s failed after %d attempts: %s", description, attempt, exc)
                raise
            delay = backoff_delay(attempt, base_s, max_s)
            if retry_after is not None:
                hinted = retry_after(exc)
                if hinted is not None:
                    delay = max(delay, hinted)
            logger.warning(
                "%s attempt %d/%d failed (%s: %s); retrying in %.1fs",
                description, attempt, max_attempts, type(exc).__name__, exc, delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")
