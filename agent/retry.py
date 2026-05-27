"""
Retry utilities — Phase 10
Exponential-backoff wrapper for Anthropic API calls.

Retries on:
  - anthropic.APIConnectionError  (network / DNS issues)
  - anthropic.RateLimitError      (HTTP 429)
  - anthropic.APIStatusError      with status_code >= 500 (server errors)

Does NOT retry on:
  - anthropic.AuthenticationError (bad key — will never succeed)
  - anthropic.BadRequestError     (malformed request — will never succeed)
  - Other 4xx errors

Default policy: 3 attempts, base delay 1 s, doubling each time (1 s, 2 s).
"""

import functools
import logging
import time
from typing import Callable, TypeVar

import anthropic

logger = logging.getLogger("lumenx.retry")

_RETRYABLE_TYPES = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
)

T = TypeVar("T")


def call_with_retry(
    fn: Callable[..., T],
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs,
) -> T:
    """
    Call *fn* with *args*/*kwargs*, retrying on transient Anthropic errors.

    Args:
        fn:           The callable to invoke (typically client.messages.create).
        max_attempts: Total number of attempts (1 = no retry).
        base_delay:   Initial sleep seconds before first retry.
        max_delay:    Upper bound on sleep seconds.
        *args/**kwargs: Passed directly to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        The last exception if all attempts are exhausted, or any
        non-retryable exception immediately.
    """
    delay = base_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)

        except _RETRYABLE_TYPES as exc:
            if attempt >= max_attempts:
                logger.error(
                    "All %d attempts exhausted (%s: %s)",
                    max_attempts, type(exc).__name__, exc,
                )
                raise
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs …",
                attempt, max_attempts, type(exc).__name__, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                if attempt >= max_attempts:
                    logger.error(
                        "All %d attempts exhausted (HTTP %d: %s)",
                        max_attempts, exc.status_code, exc,
                    )
                    raise
                logger.warning(
                    "Attempt %d/%d server error HTTP %d. Retrying in %.1fs …",
                    attempt, max_attempts, exc.status_code, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                # 4xx — bad request, auth error, etc. — raise immediately
                raise


def with_retry(
    fn: Callable | None = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """
    Decorator form of call_with_retry.

        @with_retry
        def call_llm(...): ...

        @with_retry(max_attempts=5, base_delay=2.0)
        def call_llm(...): ...
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return call_with_retry(
                f, *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                **kwargs,
            )
        return wrapper

    if fn is not None:
        # Used as bare @with_retry (no parentheses)
        return decorator(fn)
    return decorator
