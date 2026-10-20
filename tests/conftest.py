"""
Shared pytest fixtures for the Mackie test suite.

Owns: blanking the database and model settings so .env never changes a run; the throwaway
PostgreSQL database (created once per run on the server
at TEST_DATABASE_URL, migrated as an owner role and used through an app role,
as in production) and the database fixture that empties it before each test;
autouse fixtures that block database access without the database fixture,
unmocked LLM calls, HTTP requests and DNS lookups, so tests never touch the
network or real org data; the ConcurrencyProbe used to prove that work runs
in parallel; and the suite time budget from CLAUDE.md (not applied under
coverage or to the opt-in browser and deployed tests).
Does not own test logic — that lives in the individual test_*.py files.
"""

import asyncio
import gc
import os
import time
from collections.abc import Callable
from graphlib import TopologicalSorter
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm  # noqa: F401 — imported before no_real_http, since import creates an httpx client
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

# A disposable PostgreSQL server the tests may create databases and roles on, e.g.
#   docker run -d --name mackie-test-db -p 127.0.0.1:55432:5432 -e POSTGRES_PASSWORD=mackie \
#     --tmpfs /var/lib/postgresql/data postgres:17-alpine -c fsync=off
TEST_DATABASE_URL: str = os.getenv(
    "TEST_DATABASE_URL", "postgresql://postgres:mackie@127.0.0.1:55432/postgres"
)
TEST_OWNER_ROLE: str = "mackie_test_owner"
TEST_APP_ROLE: str = "mackie_test_app"
TEST_ROLE_PASSWORD: str = "mackie-test-only"
# Emptied before each test that uses the database fixture, referencing tables first: every
# table the migrations made except their own *_migrations bookkeeping. Read from the database
# by test_database, so a new table is emptied too. DELETE is ~20x faster than TRUNCATE here.
_TABLES: tuple[str, ...] = ()
# Each table in public with the other tables its foreign keys reference.
_TABLE_REFERENCES_SQL: str = r"""
SELECT c.relname,
    array(SELECT DISTINCT f.confrelid::regclass::text FROM pg_constraint f
          WHERE f.conrelid = c.oid AND f.contype = 'f' AND f.confrelid <> c.oid) AS references
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND c.relname NOT LIKE '%\_migrations'
"""
# Settings outside PROVIDERS that choose a model or reach a paid service.
_MODEL_SELECTORS: tuple[str, ...] = (
    "LLM_PROVIDER",
    "FAST_MODEL",
    "FAST_PROVIDER",
    "JUDGE_MODEL",
    "JUDGE_PROVIDER",
    "TAVILY_API_KEY",
)
# ConcurrencyProbe's wait for its expected calls; a lost asyncio.gather fails after this.
PROBE_TIMEOUT_S: float = 1.0


def pytest_configure(config: pytest.Config) -> None:
    """
    Blank the database and model settings, so .env can't change what the tests reach or see.

    No test can reach a real database, and every run picks the model CI picks (no provider
    configured), whatever keys the developer's .env holds. load_dotenv() never overrides a
    variable that is already set; tests that need a provider set it with monkeypatch.
    """
    os.environ["MACKIE_DATABASE_URL"] = ""
    os.environ["MACKIE_MIGRATION_URL"] = ""
    from agent.config import PROVIDERS

    for provider in PROVIDERS:
        for name in (
            provider.key_env,
            provider.model_env,
            provider.base_url_env,
            *provider.requires,
        ):
            if name:
                os.environ[name] = ""
    for name in _MODEL_SELECTORS:
        os.environ[name] = ""


def _tables_to_empty(conn: psycopg.Connection) -> tuple[str, ...]:
    """The tables to empty before each test, each before the tables it references."""
    references = dict(conn.execute(_TABLE_REFERENCES_SQL).fetchall())
    parents_first = TopologicalSorter(references).static_order()
    return tuple(reversed([table for table in parents_first if table in references]))


def _ensure_role(admin: psycopg.Connection, role: str) -> None:
    """Create a login role with the test password, unless it exists."""
    try:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(TEST_ROLE_PASSWORD)
            )
        )
    except psycopg.errors.DuplicateObject:
        pass  # made by an earlier or a parallel run


@pytest.fixture(scope="session")
def test_database() -> Any:
    """
    Create a database for this run, migrate it as the owner role, and yield its URLs.

    Yields a dict with the URLs "app" (the app role's, for MACKIE_DATABASE_URL), "owner"
    (for migrations) and "admin" (a superuser's), and "cleaner", an open superuser
    connection that empties the tables before each test.
    The database is dropped at the end of the run.
    """
    from agent import db

    name = f"mackie_test_{os.getpid()}"
    try:
        admin = psycopg.connect(TEST_DATABASE_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.exit(
            f"No test database server at TEST_DATABASE_URL ({e}). Start one with the docker "
            "command in tests/conftest.py, or set TEST_DATABASE_URL.",
            returncode=2,
        )
    for role in (TEST_OWNER_ROLE, TEST_APP_ROLE):
        _ensure_role(admin, role)
    admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
    admin.execute(
        sql.SQL("CREATE DATABASE {} OWNER {}").format(
            sql.Identifier(name), sql.Identifier(TEST_OWNER_ROLE)
        )
    )
    urls = {
        role: make_conninfo(TEST_DATABASE_URL, dbname=name, user=user, password=password)
        for role, user, password in (
            ("owner", TEST_OWNER_ROLE, TEST_ROLE_PASSWORD),
            ("app", TEST_APP_ROLE, TEST_ROLE_PASSWORD),
        )
    }
    urls["admin"] = make_conninfo(TEST_DATABASE_URL, dbname=name)
    asyncio.run(db.migrate(urls["owner"], TEST_APP_ROLE))
    with psycopg.connect(urls["admin"], autocommit=True) as cleaner:
        global _TABLES
        _TABLES = _tables_to_empty(cleaner)
        urls["cleaner"] = cleaner
        yield urls
    admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
    admin.close()


@pytest.fixture
async def database(test_database, monkeypatch):
    """
    An empty database for the test, reached through the app role as Mackie reaches it.

    Yields the test_database URLs. Tests share the run's connection pool; one
    that pointed MACKIE_DATABASE_URL elsewhere has its pool closed afterwards.
    """
    from agent import db

    test_database["cleaner"].execute("".join(f"DELETE FROM {table};" for table in _TABLES))
    monkeypatch.setenv("MACKIE_DATABASE_URL", test_database["app"])
    yield test_database
    if os.environ.get("MACKIE_DATABASE_URL") != test_database["app"]:
        await db.close_pool()


@pytest.fixture
async def fresh_pool(database):
    """For tests of opening the pool: none is open when the test starts, and its own is closed after."""
    from agent import db

    await db.close_pool()
    yield database
    await db.close_pool()


@pytest.fixture(autouse=True)
def no_database_without_the_db_fixture(request):
    """Fail a test that reaches the database without asking for the database fixture."""
    if "database" in request.fixturenames:
        yield
        return
    with patch(
        "agent.db.get_pool",
        side_effect=AssertionError("this test uses the database; add the database fixture"),
    ):
        yield


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


# CLAUDE.md: "The suite must stay under 30 seconds." Enforced, so a slow test can't creep in.
# CI sets MACKIE_TEST_BUDGET_S higher: shared runners vary in speed, and a run that only
# failed on a slow machine would make the check noise.
DEFAULT_SUITE_TIME_BUDGET_S: float = 30.0
SUITE_TIME_BUDGET_S: float = float(os.getenv("MACKIE_TEST_BUDGET_S") or DEFAULT_SUITE_TIME_BUDGET_S)
_suite_started: float = 0.0


def pytest_sessionstart(session: pytest.Session) -> None:
    """Note when the run started, for the time budget."""
    global _suite_started
    _suite_started = time.perf_counter()


def pytest_collection_finish(session: pytest.Session) -> None:
    """
    Move everything imported so far out of the garbage collector's way.

    LiteLLM, Chainlit and LangGraph leave hundreds of thousands of objects
    behind at import; every full collection walked them all again, about half
    a second each, several times a run. Frozen, they are never scanned.
    """
    gc.collect()
    gc.freeze()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    Fail an otherwise green run that took longer than SUITE_TIME_BUDGET_S.

    Skipped under pytest-cov: instrumentation roughly doubles the run time,
    and the budget is about the suite, not the tooling. CI runs the plain suite.
    Also skipped when opt-in tests ran (pytest -m browser, -m deployed): they
    drive a real server and browser, and are opt-in for that reason.
    """
    elapsed = time.perf_counter() - _suite_started
    measuring_coverage = session.config.pluginmanager.hasplugin("_cov")
    ran_browser_tests = any(
        item.get_closest_marker("browser") or item.get_closest_marker("deployed")
        for item in session.items
    )
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
    ) -> None:
        """result is returned to every call; respond, if given, computes it from the call's arguments."""
        self.expected = expected
        self.result = result
        self.respond = respond
        self.in_flight = 0
        self.peak = 0
        self._all_started = asyncio.Event()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        if self.in_flight >= self.expected:
            self._all_started.set()
        try:
            await asyncio.wait_for(self._all_started.wait(), PROBE_TIMEOUT_S)
        finally:
            self.in_flight -= 1
        return self.respond(*args, **kwargs) if self.respond else self.result


@pytest.fixture
def concurrency_probe() -> Callable[..., ConcurrencyProbe]:
    """Factory for ConcurrencyProbe: concurrency_probe(expected, result) or (expected, respond=fn)."""
    return ConcurrencyProbe
