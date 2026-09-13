from resilience.retries import TransientError, retry_with_backoff
from resilience.timeouts import GlobalTimeout, NodeTimeout, run_timeout_demos


def test_retry_recovers():
    calls = {"n": 0}

    def flaky(attempt):
        calls["n"] += 1
        if attempt < 3:
            raise TransientError("fail")
        return "ok"

    out = retry_with_backoff(flaky, max_attempts=3, initial=0.01, max_interval=0.05, jitter=0.0)
    assert out["value"] == "ok"
    assert out["attempts"] == 3
    assert calls["n"] == 3


def test_timeouts_fire():
    out = run_timeout_demos()
    assert out["node_timeout"]["fired"] is True
    assert out["global_timeout"]["fired"] is True
    assert "exceeded" in out["node_timeout"]["error"]
    assert "exceeded" in out["global_timeout"]["error"]
