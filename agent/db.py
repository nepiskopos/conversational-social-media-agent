"""
PostgreSQL access for Mackie: the connection pool, org-scoped transactions and migrations.

Owns: the connection pool (one per event loop, opened on first use, with a
timeout on every connect, statement, lock wait and idle transaction), the
checks made when it opens (the app's role must be subject to row-level
security, and the schema must be up to date), org_transaction(), which scopes
a transaction to one org so the database itself refuses any other org's rows,
and migrate(), which applies migrations/*.sql as the owner role and grants the
app's role only what it needs.
Does not own what is stored (memory.py, accounts.py) or conversation
checkpoints (graph.py).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from psycopg import AsyncConnection, sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

MIGRATIONS_DIR: Path = Path(__file__).parent.parent / "migrations"
ORG_ID_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DEFAULT_POOL_SIZE: int = 10
# Nothing waits forever: a connect, a statement, a row-lock wait and a transaction left
# open by a crashed caller each have a ceiling.
CONNECT_TIMEOUT_S: int = 10
POOL_TIMEOUT_S: float = 30.0
STATEMENT_TIMEOUT_MS: int = 30_000
LOCK_TIMEOUT_MS: int = 10_000
IDLE_IN_TRANSACTION_TIMEOUT_MS: int = 60_000
# The row-level security policies in migrations/ compare each row's org with this setting.
ORG_SETTING: str = "app.org_id"
SET_ORG_SQL: str = f"SELECT set_config('{ORG_SETTING}', %s, %s)"
# Any constant works; it only has to be the same for every migrate() run.
MIGRATION_LOCK_KEY: int = 7_351_002
# Run on every migrate(), so tables added later (e.g. by a LangGraph upgrade) are covered.
# The app's role reads and writes rows; it can't change the schema or the migration records.
_GRANTS_SQL: str = """
GRANT USAGE ON SCHEMA public TO {role};
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role};
REVOKE INSERT, UPDATE, DELETE ON schema_migrations, checkpoint_migrations FROM {role};
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role};
"""
_pools: dict[asyncio.AbstractEventLoop, asyncio.Future[AsyncConnectionPool]] = {}
logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """The database is not set up for Mackie; the message says what to do."""


def check_org_id(org_id: str) -> str:
    """
    Return org_id if it is a slugify() id, so it can name a thread or an org's rows.

    Raises:
        ValueError: any other id.
    """
    if not ORG_ID_PATTERN.fullmatch(org_id):
        raise ValueError(f"invalid org id {org_id!r}; org ids come from slugify()")
    return org_id


def database_url() -> str:
    """
    The app's connection string, from MACKIE_DATABASE_URL.

    Raises:
        DatabaseError: MACKIE_DATABASE_URL is not set.
    """
    url = os.getenv("MACKIE_DATABASE_URL", "").strip()
    if not url:
        raise DatabaseError(
            "MACKIE_DATABASE_URL is not set. Point it at Mackie's PostgreSQL database "
            "(see Configuration in the README)."
        )
    return url


def migration_files() -> list[Path]:
    """The migrations in the order they apply: 001_name.sql, 002_name.sql, ..."""
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))


def _migration_version(path: Path) -> int:
    """The number a migration file starts with."""
    return int(path.name.split("_", 1)[0])


async def _check_database(pool: AsyncConnectionPool) -> None:
    """
    Refuse a role that row-level security doesn't apply to, or an out-of-date schema.

    Raises:
        DatabaseError: with what to change.
    """
    async with pool.connection() as conn:
        role = await (
            await conn.execute(
                "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )
        ).fetchone()
        if role and (role["rolsuper"] or role["rolbypassrls"]):
            raise DatabaseError(
                f"MACKIE_DATABASE_URL connects as {role['rolname']}, a role that bypasses row-level "
                "security, so one org could read another's data. Connect as the app role "
                "(see Configuration in the README)."
            )
        exists = await (
            await conn.execute("SELECT to_regclass('schema_migrations') IS NOT NULL AS ok")
        ).fetchone()
        applied = 0
        if exists and exists["ok"]:
            row = await (
                await conn.execute("SELECT coalesce(max(version), 0) AS v FROM schema_migrations")
            ).fetchone()
            applied = row["v"] if row else 0
    expected = max((_migration_version(path) for path in migration_files()), default=0)
    if applied < expected:
        raise DatabaseError(
            f"The database schema is at version {applied}; this Mackie needs {expected}. "
            "Run: python scripts/migrate.py"
        )


def _pool_size() -> int:
    """
    MACKIE_DATABASE_POOL_SIZE, or DEFAULT_POOL_SIZE when unset.

    Raises:
        DatabaseError: it is set to something other than a whole number above 0.
    """
    setting = os.getenv("MACKIE_DATABASE_POOL_SIZE", "").strip()
    if not setting:
        return DEFAULT_POOL_SIZE
    if not setting.isdigit() or int(setting) < 1:
        raise DatabaseError(
            f"MACKIE_DATABASE_POOL_SIZE={setting!r} must be a whole number of connections, e.g. 10."
        )
    return int(setting)


async def _open_pool() -> AsyncConnectionPool:
    """Open and check a pool for the running event loop."""
    pool: AsyncConnectionPool = AsyncConnectionPool(
        database_url(),
        min_size=1,
        max_size=_pool_size(),
        timeout=POOL_TIMEOUT_S,
        open=False,
        kwargs={
            # What LangGraph's checkpointer needs; memory.py opens its own transactions.
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": CONNECT_TIMEOUT_S,
            "application_name": "mackie",
            "options": (
                f"-c statement_timeout={STATEMENT_TIMEOUT_MS} -c lock_timeout={LOCK_TIMEOUT_MS} "
                f"-c idle_in_transaction_session_timeout={IDLE_IN_TRANSACTION_TIMEOUT_MS}"
            ),
        },
    )
    await pool.open(wait=True, timeout=POOL_TIMEOUT_S)
    try:
        await _check_database(pool)
    except BaseException:
        await pool.close()
        raise
    return pool


async def get_pool() -> AsyncConnectionPool:
    """
    The connection pool for the running event loop, opened and checked on first use.

    A pool belongs to the event loop that opened it, and scripts and tests run
    several loops one after another, hence one per loop. Sessions that start
    together share one opening; a failed opening is retried by the next caller.

    Raises:
        DatabaseError: MACKIE_DATABASE_URL is not set, the role bypasses row-level
            security, or the schema needs migrating.
    """
    loop = asyncio.get_running_loop()
    for closed in [other for other in _pools if other.is_closed()]:
        del _pools[closed]
    if loop not in _pools:
        _pools[loop] = asyncio.ensure_future(_open_pool())
    try:
        return await asyncio.shield(_pools[loop])
    except BaseException:
        if _pools.get(loop) is not None and _pools[loop].done():
            del _pools[loop]
        raise


async def close_pool() -> None:
    """Close the running event loop's pool, if one is open. Scripts call this before exiting."""
    opening = _pools.pop(asyncio.get_running_loop(), None)
    if opening is not None and opening.done() and not opening.exception():
        await opening.result().close()


@asynccontextmanager
async def _connection() -> AsyncIterator[AsyncConnection[DictRow]]:
    """
    A pooled connection; when one turns out to be dead, every idle one is checked.

    After a database restart or failover each pooled connection is dead, and
    without the check each would fail one request in turn. Checking every
    checkout instead would cost a round trip per transaction.
    """
    pool = await get_pool()
    connection_lost = False
    try:
        async with pool.connection() as conn:
            try:
                yield conn
            except psycopg.OperationalError:
                # Not for a timeout, a lock wait or a deadlock: the connection is fine then,
                # and checking the whole pool would make every other request wait.
                connection_lost = conn.broken
                raise
    finally:
        if connection_lost:
            logger.warning("Lost a database connection; checking the others")
            await pool.check()


@asynccontextmanager
async def org_transaction(
    org_id: str, *, read_only: bool = False
) -> AsyncIterator[AsyncConnection[DictRow]]:
    """
    Run a transaction that can see and change only this org's rows.

    The org setting is local to the transaction, so it ends with it. Every
    use of a pooled connection goes through this or transaction(), which set
    the org they need, never pool.connection() directly; the one exception is
    graph.OrgCheckpointer, which sets its org on every cursor. A read-only
    transaction reads one snapshot, so a profile assembled from several tables
    is never half of another session's write.

    Raises:
        ValueError: org_id is not a slugify() id.
    """
    check_org_id(org_id)
    async with _connection() as conn, conn.transaction():
        if read_only:
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        await conn.execute(SET_ORG_SQL, (org_id, True))
        yield conn


@asynccontextmanager
async def transaction() -> AsyncIterator[AsyncConnection[DictRow]]:
    """
    A transaction outside any org, for accounts; org tables show it no rows.

    The org is cleared for the transaction because a pooled connection keeps
    the org a checkpointer last set on it (graph.OrgCheckpointer), and
    memberships, which accounts read, are not filtered by org.
    """
    async with _connection() as conn, conn.transaction():
        await conn.execute(SET_ORG_SQL, ("", True))
        yield conn


async def migrate(owner_url: str, app_role: str) -> list[str]:
    """
    Apply the migrations not yet applied, as the owner role, and grant the app's role its rights.

    LangGraph's checkpoint tables come first, since the first migration puts
    row-level security on them. Concurrent runs wait for each other.

    Args:
        owner_url: Connection string of the role that owns the schema.
        app_role:  The role the app connects as (the user in MACKIE_DATABASE_URL).

    Returns:
        The names of the migration files applied by this run.
    """
    # Imported here: only migrations need it, and graph.py imports this module.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    applied: list[str] = []
    async with await AsyncConnection.connect(
        owner_url, autocommit=True, row_factory=dict_row
    ) as conn:
        await conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
        await AsyncPostgresSaver(conn).setup()
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version integer PRIMARY KEY, name text NOT NULL, "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        rows = await (await conn.execute("SELECT version FROM schema_migrations")).fetchall()
        done = {row["version"] for row in rows}
        for path in migration_files():
            if _migration_version(path) in done:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text())
                await conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                    (_migration_version(path), path.name),
                )
            applied.append(path.name)
        await conn.execute(sql.SQL(_GRANTS_SQL).format(role=sql.Identifier(app_role)))
        await conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
    return applied


def role_of(url: str) -> str:
    """
    The role a connection string logs in as.

    Raises:
        DatabaseError: the string names no user.
    """
    user = conninfo_to_dict(url).get("user")
    if not user:
        raise DatabaseError("MACKIE_DATABASE_URL must name the user Mackie connects as.")
    return str(user)
