"""Smoke tests for the core package."""

from churn_platform import health_check


def test_health_check() -> None:
    """The package can be imported and reports a healthy status."""
    assert health_check() == "ok"
