import random
import time

from config import (
    RETRY_INITIAL_INTERVAL,
    RETRY_JITTER,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_INTERVAL,
)


class TransientError(Exception):
    pass


def retry_with_backoff(
    fn,
    max_attempts=RETRY_MAX_ATTEMPTS,
    initial=RETRY_INITIAL_INTERVAL,
    max_interval=RETRY_MAX_INTERVAL,
    jitter=RETRY_JITTER,
):
    delay = initial
    last = None
    log = []
    for attempt in range(1, max_attempts + 1):
        try:
            value = fn(attempt)
            log.append({"attempt": attempt, "ok": True, "waited": 0})
            return {"value": value, "attempts": attempt, "log": log}
        except TransientError as exc:
            last = exc
            if attempt == max_attempts:
                log.append({"attempt": attempt, "ok": False, "error": str(exc)})
                raise
            wait = min(delay, max_interval) + random.uniform(0, jitter)
            log.append({"attempt": attempt, "ok": False, "waited": round(wait, 3), "error": str(exc)})
            time.sleep(wait)
            delay *= 2
    raise last


def simulate_transient(attempt):
    if attempt < 3:
        raise TransientError("simulated gateway timeout on attempt %d" % attempt)
    return {"recovered": True, "attempt": attempt}


def run_retry_demo():
    return retry_with_backoff(simulate_transient)
