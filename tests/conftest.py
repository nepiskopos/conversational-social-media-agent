"""
Shared pytest fixtures for the Mackie test suite.

Owns: the tmp_data_dir fixture that redirects disk I/O to a temp directory,
autouse fixtures that block unmocked LLM calls, HTTP requests and DNS
lookups, so tests never touch the network, real org data or the production
SQLite checkpoint database; the ConcurrencyProbe used to prove that work
runs in parallel; and the suite time budget from CLAUDE.md (not applied under
coverage or to the browser tests).
Does not own test logic — that lives in the individual test_*.py files.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm  # noqa: F401 — imported before no_real_http, since import creates an httpx client
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """
    Redirect DATA_DIR and CHECKPOINTS_DB to temp paths so tests never touch
    real org data or the real SQLite checkpoint database.
    """
    with (
        patch("agent.memory.DATA_DIR", tmp_path),
        patch("agent.graph.CHECKPOINTS_DB", tmp_path / "checkpoints.db"),
    ):
        yield tmp_path


@pytest.fixture(autouse=True)
def no_unmocked_llm_calls():
    """
    Fail any LLM call a test did not mock, keeping the suite offline.

    Tests that expect a call patch agent.<module>.litellm.acompletion
    themselves; that inner patch takes precedence over this one.
    """
    with patch("litellm.acompletion", side_effect=AssertionError("unmocked LLM call")):
        yield


@pytest.fixture(autouse=True)
def no_real_dns():
    """
    Resolve every hostname to a public address (example.com's), keeping the suite offline.

    The SSRF guard looks up each host before fetching; tests that need another
    answer patch agent.research._resolve_host themselves.
    """
    with patch("agent.research._resolve_host", new=AsyncMock(return_value=["93.184.216.34"])):
        yield


@pytest.fixture(autouse=True)
def no_real_http():
    """
    Fail any website fetch a test did not mock, keeping the suite offline.

    Patches httpx.AsyncClient for every caller (research.py and the Tavily
    client both use httpx). Tests that fetch pages patch
    agent.research.httpx.AsyncClient themselves; that inner patch takes
    precedence over this one.
    """
    with patch("httpx.AsyncClient") as client_class:
        client = client_class.return_value.__aenter__.return_value
        client.get = client.post = client.request = AsyncMock(
            side_effect=AssertionError("unmocked HTTP request")
        )
        yield


# CLAUDE.md: "The suite must stay under 10 seconds." Enforced, so a slow test can't creep in.
SUITE_TIME_BUDGET_S: float = 10.0
_suite_started: float = 0.0


def pytest_sessionstart(session: pytest.Session) -> None:
    """Note when the run started, for the time budget."""
    global _suite_started
    _suite_started = time.perf_counter()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    Fail an otherwise green run that took longer than SUITE_TIME_BUDGET_S.

    Skipped under pytest-cov: instrumentation roughly doubles the run time,
    and the budget is about the suite, not the tooling. CI runs the plain suite.
    Also skipped when browser tests ran (pytest -m browser): they start a real
    server and browser, and are opt-in for that reason.
    """
    elapsed = time.perf_counter() - _suite_started
    measuring_coverage = session.config.pluginmanager.hasplugin("_cov")
    ran_browser_tests = any(item.get_closest_marker("browser") for item in session.items)
    over_budget = elapsed > SUITE_TIME_BUDGET_S
    if exitstatus == 0 and over_budget and not (measuring_coverage or ran_browser_tests):
        print(f"\nSuite took {elapsed:.1f}s; the budget is {SUITE_TIME_BUDGET_S:.0f}s.")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


class ConcurrencyProbe:
    """
    Async stand-in for an I/O call that returns only once `expected` calls are in flight.

    Code that runs the calls in parallel (asyncio.gather) gets all of them
    started together and they finish at once. Code that awaits them one by one
    never reaches `expected`, so the first call times out — a lost
    asyncio.gather fails the test instead of just making production slower.
    No wall-clock timing is involved, so the check doesn't flake on a slow CI runner.
    """

    def __init__(
        self,
        expected: int,
        result: Any = None,
        respond: Callable[..., Any] | None = None,
        timeout_s: float = 1.0,
    ) -> None:
        """result is returned to every call; respond, if given, computes it from the call's arguments."""
        self.expected = expected
        self.result = result
        self.respond = respond
        self.timeout_s = timeout_s
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self._all_started = asyncio.Event()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        if self.in_flight >= self.expected:
            self._all_started.set()
        try:
            await asyncio.wait_for(self._all_started.wait(), self.timeout_s)
        finally:
            self.in_flight -= 1
        return self.respond(*args, **kwargs) if self.respond else self.result


@pytest.fixture
def concurrency_probe() -> Callable[..., ConcurrencyProbe]:
    """Factory for ConcurrencyProbe: concurrency_probe(expected, result) or (expected, respond=fn)."""
    return ConcurrencyProbe
