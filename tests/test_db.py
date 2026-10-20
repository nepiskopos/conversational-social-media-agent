"""
Unit tests for agent/db.py: the connection string and role_of, org ids, the
pool (checked when it opens, with timeouts, retried after a failed opening,
recovering when the database drops its connections; sharing one opening
between sessions is owned by test_sessions.py), org-scoped and read-only
transactions, rollback on failure, and migrations. Row-level security as an attacker meets it, and
what the app's role may not do, are owned by test_security.py.
"""

import asyncio
import logging
import os
import time
from unittest.mock import AsyncMock, patch

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool

# Every connection the pool opens has application_name 'mackie' (agent.db._open_pool).
_KILL_POOL_SQL = (
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
    "WHERE application_name = 'mackie' AND datname = current_database()"
)


def test_database_url_explains_a_missing_setting(monkeypatch):
    from agent.db import DatabaseError, database_url

    monkeypatch.setenv("MACKIE_DATABASE_URL", "  ")
    with pytest.raises(DatabaseError, match="MACKIE_DATABASE_URL is not set"):
        database_url()
    monkeypatch.setenv("MACKIE_DATABASE_URL", "postgresql://app@db/mackie ")
    assert database_url() == "postgresql://app@db/mackie"


@pytest.mark.parametrize("org_id", ["", "../etc", "Org", "a b", "org:1", "-org", "org-"])
def test_check_org_id_refuses_anything_but_a_slug(org_id):
    from agent.db import check_org_id

    with pytest.raises(ValueError, match="invalid org id"):
        check_org_id(org_id)


def test_check_org_id_returns_a_slug():
    from agent.db import check_org_id

    assert check_org_id("brca-strong-2") == "brca-strong-2"


def test_role_of_reads_the_user_from_a_connection_string():
    from agent.db import DatabaseError, role_of

    assert role_of("postgresql://mackie_app:secret@db:5432/mackie") == "mackie_app"
    assert role_of("host=db user=owner dbname=mackie") == "owner"
    with pytest.raises(DatabaseError, match="must name the user"):
        role_of("postgresql://db/mackie")


def test_migration_files_are_numbered_and_in_order():
    from agent.db import migration_files

    names = [path.name for path in migration_files()]
    assert names[0] == "001_initial.sql"
    assert names == sorted(names)


async def test_the_pool_sets_timeouts_on_every_connection(database):
    from agent import db

    async with db.transaction() as conn:
        row = await (
            await conn.execute(
                "SELECT current_setting('statement_timeout') AS s, "
                "current_setting('lock_timeout') AS l, "
                "current_setting('idle_in_transaction_session_timeout') AS i"
            )
        ).fetchone()
    assert row == {"s": "30s", "l": "10s", "i": "1min"}


async def test_a_superuser_role_is_refused(fresh_pool, monkeypatch):
    """A superuser bypasses row-level security, so one org could read another's rows."""
    from agent import db

    monkeypatch.setenv("MACKIE_DATABASE_URL", fresh_pool["admin"])
    with pytest.raises(db.DatabaseError, match="bypasses row-level security"):
        await db.get_pool()


async def test_a_role_with_bypassrls_is_refused(fresh_pool, monkeypatch):
    """BYPASSRLS without superuser is the same hole, so it is refused too."""
    from agent import db

    from .conftest import TEST_ROLE_PASSWORD

    cleaner = fresh_pool["cleaner"]
    role = f"mackie_test_bypass_{os.getpid()}"
    cleaner.execute(
        sql.SQL("CREATE ROLE {} LOGIN BYPASSRLS PASSWORD {}").format(
            sql.Identifier(role), sql.Literal(TEST_ROLE_PASSWORD)
        )
    )
    try:
        url = make_conninfo(fresh_pool["app"], user=role, password=TEST_ROLE_PASSWORD)
        monkeypatch.setenv("MACKIE_DATABASE_URL", url)
        with pytest.raises(db.DatabaseError, match=f"connects as {role}.*bypasses row-level"):
            await db.get_pool()
    finally:
        cleaner.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.mark.parametrize("pool_size", ["ten", "0"])
async def test_a_pool_size_that_isnt_a_count_is_refused(fresh_pool, monkeypatch, pool_size):
    from agent import db

    monkeypatch.setenv("MACKIE_DATABASE_POOL_SIZE", pool_size)
    with pytest.raises(db.DatabaseError, match=f"MACKIE_DATABASE_POOL_SIZE='{pool_size}'"):
        await db.get_pool()


async def test_a_failed_pool_opening_is_retried_by_the_next_caller(fresh_pool):
    """A database still starting up must not leave the app failing until it restarts."""
    from agent import db

    original_open = db._open_pool
    attempts = []

    async def flaky_open():
        attempts.append(1)
        if len(attempts) == 1:
            raise db.DatabaseError("database starting up")
        return await original_open()

    with patch.object(db, "_open_pool", flaky_open):
        with pytest.raises(db.DatabaseError, match="starting up"):
            await db.get_pool()
        pool = await db.get_pool()
    assert len(attempts) == 2
    assert await db.get_pool() is pool


async def test_an_out_of_date_schema_is_refused_until_migrated(fresh_pool, tmp_path):
    from agent import db

    for name in ("001_initial.sql", "002_next.sql"):
        (tmp_path / name).write_text("SELECT 1;")
    with (
        patch("agent.db.MIGRATIONS_DIR", tmp_path),
        pytest.raises(db.DatabaseError, match="version 1; this Mackie needs 2.*migrate.py"),
    ):
        await db.get_pool()


async def test_close_pool_without_a_pool_does_nothing(fresh_pool):
    from agent import db

    await db.close_pool()


async def test_org_transaction_is_scoped_to_its_org_and_ends_with_it(database):
    from agent import db

    async with db.org_transaction("org-a") as conn:
        row = await (await conn.execute("SELECT current_setting('app.org_id') AS org")).fetchone()
        assert row["org"] == "org-a"
    async with db.transaction() as conn:
        row = await (
            await conn.execute("SELECT current_setting('app.org_id', true) AS org")
        ).fetchone()
        assert not row["org"]


async def test_a_read_only_org_transaction_refuses_writes(database):
    from agent import db

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        async with db.org_transaction("org-a", read_only=True) as conn:
            await conn.execute("INSERT INTO orgs (org_id) VALUES ('org-a')")


async def test_migrate_again_applies_nothing_and_keeps_the_grants(fresh_pool):
    """The test database is already migrated; a second run (e.g. a redeploy) must be a no-op."""
    from agent import db, memory

    from .conftest import TEST_APP_ROLE

    assert await db.migrate(fresh_pool["owner"], TEST_APP_ROLE) == []
    # The pool opens after the migration, so its checks run as the app role against it.
    assert await db.get_pool() is not None
    await memory.fill_org_details("org-a", "Still writable")
    assert (await memory.load_profile("org-a"))["name"] == "Still writable"


async def test_a_failing_migration_leaves_nothing_behind(database, tmp_path):
    from agent import db

    from .conftest import TEST_APP_ROLE

    (tmp_path / "001_initial.sql").write_text((db.MIGRATIONS_DIR / "001_initial.sql").read_text())
    (tmp_path / "002_bad.sql").write_text("CREATE TABLE half_done (x int); SELECT 1/0;")
    with (
        patch("agent.db.MIGRATIONS_DIR", tmp_path),
        pytest.raises(psycopg.errors.DivisionByZero),
    ):
        await db.migrate(database["owner"], TEST_APP_ROLE)
    cleaner = database["cleaner"]
    assert cleaner.execute("SELECT to_regclass('half_done')").fetchone() == (None,)
    assert cleaner.execute("SELECT max(version) FROM schema_migrations").fetchone() == (1,)


async def test_after_the_database_drops_its_connections_only_one_request_fails(
    fresh_pool, monkeypatch, caplog
):
    """A restart kills every pooled connection; the first failure checks them all, and says so."""
    from agent import db, memory

    pool_size = 4
    monkeypatch.setenv("MACKIE_DATABASE_POOL_SIZE", str(pool_size))
    all_in = asyncio.Barrier(pool_size)

    async def hold_a_connection() -> None:
        async with db.org_transaction("org-a"):
            await all_in.wait()

    await asyncio.wait_for(asyncio.gather(*[hold_a_connection() for _ in range(pool_size)]), 5)
    killed = fresh_pool["cleaner"].execute(_KILL_POOL_SQL).fetchall()
    assert len(killed) == pool_size

    failures = 0
    with (
        patch.object(
            AsyncConnectionPool, "check", autospec=True, side_effect=AsyncConnectionPool.check
        ) as check,
        caplog.at_level(logging.WARNING, logger="agent.db"),
    ):
        for name in ("first", "second"):
            try:
                await memory.fill_org_details("org-a", name)
            except psycopg.OperationalError:
                failures += 1
    assert failures == 1
    assert check.await_count == failures
    lost = [r.getMessage() for r in caplog.records if r.name == "agent.db"]
    assert lost == ["Lost a database connection; checking the others"] * failures
    for _ in range(4):
        assert (await memory.load_profile("org-a"))["name"] in ("first", "second")
    profiles = await asyncio.gather(*[memory.load_profile("org-a") for _ in range(4)])
    assert all(profile["org_id"] == "org-a" for profile in profiles)


async def test_a_slow_statement_and_a_held_lock_fail_instead_of_hanging(fresh_pool, monkeypatch):
    """Neither breaks the connection, so neither makes every other request wait for a pool check."""
    from agent import db, memory

    check = AsyncMock()
    monkeypatch.setattr(AsyncConnectionPool, "check", check)

    # The lock wait is a statement too, so its own, shorter timeout must be the one that fires.
    monkeypatch.setattr(db, "STATEMENT_TIMEOUT_MS", 150)
    monkeypatch.setattr(db, "LOCK_TIMEOUT_MS", 50)
    started = time.monotonic()
    with pytest.raises(psycopg.errors.QueryCanceled):
        async with db.transaction() as conn:
            await conn.execute("SELECT pg_sleep(2)")
    assert time.monotonic() - started < 1.5

    cleaner = fresh_pool["cleaner"]
    cleaner.execute("INSERT INTO orgs (org_id) VALUES ('org-a')")
    with cleaner.transaction():
        cleaner.execute("SELECT 1 FROM orgs WHERE org_id = 'org-a' FOR UPDATE")
        started = time.monotonic()
        with pytest.raises(psycopg.errors.LockNotAvailable):
            await memory.fill_org_details("org-a", "Blocked")
        assert time.monotonic() - started < 1.5
    await memory.fill_org_details("org-a", "Unblocked")
    assert (await memory.load_profile("org-a"))["name"] == "Unblocked"
    check.assert_not_awaited()


async def test_a_failed_transaction_rolls_back_and_returns_its_connection(fresh_pool, monkeypatch):
    """With one connection, a leak would make the next transaction wait for the pool timeout."""
    from agent import db

    monkeypatch.setenv("MACKIE_DATABASE_POOL_SIZE", "1")
    monkeypatch.setattr(db, "POOL_TIMEOUT_S", 2.0)
    fresh_pool["cleaner"].execute("INSERT INTO orgs (org_id, name) VALUES ('org-a', 'Original')")
    for _ in range(30):
        with pytest.raises(RuntimeError, match="tool failed"):
            async with db.org_transaction("org-a") as conn:
                await conn.execute("UPDATE orgs SET name = 'Changed' WHERE org_id = 'org-a'")
                raise RuntimeError("tool failed")
    assert fresh_pool["cleaner"].execute("SELECT name FROM orgs").fetchall() == [("Original",)]
    async with db.org_transaction("org-a", read_only=True) as conn:
        row = await (await conn.execute("SELECT name FROM orgs")).fetchone()
    assert row == {"name": "Original"}
