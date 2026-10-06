"""
Shared pytest fixtures for the Mackie test suite.

Owns: the tmp_data_dir fixture that redirects disk I/O to a temp directory,
plus autouse fixtures that block unmocked LLM calls, HTTP requests and DNS
lookups, so tests never touch the network, real org data or the production
SQLite checkpoint database.
Does not own test logic — that lives in the individual test_*.py files.
"""

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
