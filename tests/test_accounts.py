"""
Unit tests for agent/accounts.py: password hashing, usernames, creating,
changing, listing and deleting accounts (ensure_user included), logging in
(wrong passwords, unknown users, the lockout) and org memberships (open_org,
grant, revoke, has_access). The login screens are owned by test_ui.py; the admin commands
by test_admin_scripts.py.
"""

import asyncio
from datetime import timedelta
from unittest.mock import patch

import pytest


def test_a_password_hash_verifies_only_its_own_password():
    from agent.accounts import hash_password, verify_password

    stored = hash_password("correct horse battery")
    assert stored.startswith("scrypt$16$8$1$")
    assert verify_password("correct horse battery", stored)
    assert not verify_password("correct horse batterY", stored)


def test_the_same_password_hashes_differently_each_time():
    """A random salt per password, so equal passwords can't be spotted in the table."""
    from agent.accounts import hash_password

    assert hash_password("same password!") != hash_password("same password!")


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "plain-text-password",
        "md5$x$y",
        "scrypt$16$8$1$not base64!$x",
        "bcrypt$16$8$1$AA==$AA==",
    ],
)
def test_verify_password_refuses_anything_but_a_scrypt_hash(stored):
    from agent.accounts import verify_password

    assert not verify_password("plain-text-password", stored)


@pytest.mark.parametrize("username", ["", "a b", "x" * 65, "ünïcode", "semi;colon"])
def test_normalize_username_refuses_odd_names(username):
    from agent.accounts import AccountError, normalize_username

    with pytest.raises(AccountError, match="1 to 64 characters"):
        normalize_username(username)


def test_normalize_username_ignores_case_and_spaces():
    from agent.accounts import normalize_username

    assert normalize_username("  Alice@Example.ORG ") == "alice@example.org"


async def test_create_user_then_log_in(database):
    from agent import accounts

    user = await accounts.create_user("Alice@Example.org", "a long passphrase")
    assert user.username == "alice@example.org"
    assert await accounts.authenticate("ALICE@example.org", "a long passphrase") == user
    assert await accounts.get_user(user.user_id) == user


async def test_create_user_refuses_a_taken_name(database):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    with pytest.raises(accounts.AccountError, match="taken"):
        await accounts.create_user("ALICE", "another passphrase")


@pytest.mark.parametrize("writer", ["create_user", "ensure_user", "set_password"])
async def test_every_account_writer_refuses_a_short_password(database, writer):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    short = "x" * (accounts.MIN_PASSWORD_CHARS - 1)
    with pytest.raises(accounts.AccountError, match=f"at least {accounts.MIN_PASSWORD_CHARS}"):
        await getattr(accounts, writer)("bob" if writer == "create_user" else "alice", short)
    assert await accounts.authenticate("alice", "a long passphrase") is not None
    assert [name for name, _ in await accounts.list_users()] == ["alice"]


async def test_the_password_is_never_stored_as_typed(database):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    (stored,) = database["cleaner"].execute("SELECT password_hash FROM users").fetchone()
    assert "a long passphrase" not in stored
    assert stored.startswith("scrypt$")


async def test_an_unknown_user_or_a_locked_account_costs_one_password_check_too(database):
    """Otherwise a fast "no" would tell an attacker which usernames exist, or are locked."""
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    database["cleaner"].execute("UPDATE users SET locked_until = now() + interval '1 hour'")
    with patch("agent.accounts.verify_password", return_value=True) as verify:
        assert await accounts.authenticate("nobody", "a long passphrase") is None
        assert await accounts.authenticate("not a name!", "a long passphrase") is None
        assert await accounts.authenticate("alice", "a long passphrase") is None
    assert verify.call_count == 3


def test_the_unknown_user_hash_costs_as_much_as_a_real_one():
    """A cheaper dummy hash would make a wrong username answer faster than a wrong password."""
    from agent import accounts

    accounts._unknown_user_hash.cache_clear()
    try:
        cost = accounts._unknown_user_hash().split("$")[1:4]
        assert cost == [str(accounts.SCRYPT_N), str(accounts.SCRYPT_R), str(accounts.SCRYPT_P)]
        assert cost == accounts.hash_password("a long passphrase").split("$")[1:4]
    finally:
        accounts._unknown_user_hash.cache_clear()


async def test_repeated_wrong_passwords_lock_the_account_for_its_full_length(database):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    for _ in range(accounts.MAX_FAILED_LOGINS):
        assert await accounts.authenticate("alice", "wrong passphrase") is None
    (locked_for,) = database["cleaner"].execute("SELECT locked_until - now() FROM users").fetchone()
    assert accounts.LOCKOUT - timedelta(seconds=5) < locked_for <= accounts.LOCKOUT
    # Locked: even the right password is refused, with the same answer as a wrong one.
    assert await accounts.authenticate("alice", "a long passphrase") is None
    database["cleaner"].execute("UPDATE users SET locked_until = now() - interval '1 second'")
    assert await accounts.authenticate("alice", "a long passphrase") is not None


async def test_a_right_password_resets_the_count_of_wrong_ones(database):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    for _ in range(accounts.MAX_FAILED_LOGINS - 1):
        await accounts.authenticate("alice", "wrong passphrase")
    assert await accounts.authenticate("alice", "a long passphrase") is not None
    await accounts.authenticate("alice", "wrong passphrase")
    assert await accounts.authenticate("alice", "a long passphrase") is not None


async def test_concurrent_wrong_passwords_are_all_counted(database):
    """Each wrong password is counted by one atomic UPDATE, so parallel guesses can't skip one."""
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    await asyncio.gather(
        *[
            accounts.authenticate("alice", "wrong passphrase")
            for _ in range(accounts.MAX_FAILED_LOGINS)
        ]
    )
    assert await accounts.authenticate("alice", "a long passphrase") is None


async def test_a_right_password_is_refused_if_guesses_locked_the_account_meanwhile(database):
    """The check runs with no row lock held, so the count clears only if still unlocked."""
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    real_verify = accounts._verify

    async def guesses_lock_it_during_the_check(password: str, stored: str | None) -> bool:
        right = await real_verify(password, stored)
        database["cleaner"].execute("UPDATE users SET locked_until = now() + interval '1 hour'")
        return right

    with patch("agent.accounts._verify", guesses_lock_it_during_the_check):
        assert await accounts.authenticate("alice", "a long passphrase") is None
    (still_locked,) = (
        database["cleaner"].execute("SELECT locked_until > now() FROM users").fetchone()
    )
    assert still_locked


async def test_set_password_changes_it_and_lifts_a_lockout(database):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    for _ in range(accounts.MAX_FAILED_LOGINS):
        await accounts.authenticate("alice", "wrong passphrase")
    assert await accounts.set_password("Alice", "a brand new passphrase")
    assert await accounts.authenticate("alice", "a long passphrase") is None
    assert await accounts.authenticate("alice", "a brand new passphrase") is not None


async def test_set_password_for_an_unknown_user_changes_nothing(database):
    from agent import accounts

    assert not await accounts.set_password("nobody", "a long passphrase")
    assert await accounts.list_users() == []


async def test_delete_user_removes_the_login_and_memberships(database):
    from agent import accounts

    user = await accounts.create_user("alice", "a long passphrase")
    await accounts.grant("alice", "org-a")
    assert await accounts.delete_user("ALICE")
    assert await accounts.get_user(user.user_id) is None
    assert await accounts.authenticate("alice", "a long passphrase") is None
    assert database["cleaner"].execute("SELECT count(*) FROM memberships").fetchone() == (0,)
    assert not await accounts.delete_user("alice")


async def test_a_user_opens_a_new_org_and_becomes_its_member(database):
    from agent import accounts

    alice = await accounts.create_user("alice", "a long passphrase")
    assert await accounts.open_org(alice.user_id, "org-a")
    assert await accounts.user_orgs(alice.user_id) == ["org-a"]
    assert await accounts.open_org(alice.user_id, "org-a")


@pytest.mark.parametrize("setting", ["false", "False", "0", "no", "off"])
async def test_only_admins_create_orgs_when_users_may_not(database, monkeypatch, setting):
    """With MACKIE_USERS_CREATE_ORGS=false, nobody claims an org before its team arrives."""
    from agent import accounts

    monkeypatch.setenv("MACKIE_USERS_CREATE_ORGS", setting)
    alice = await accounts.create_user("alice", "a long passphrase")
    assert not await accounts.open_org(alice.user_id, "org-a")
    assert await accounts.list_users() == [("alice", [])]
    await accounts.grant("alice", "org-a")
    assert await accounts.open_org(alice.user_id, "org-a")


async def test_a_user_cannot_open_someone_elses_org(database):
    from agent import accounts

    alice = await accounts.create_user("alice", "a long passphrase")
    bob = await accounts.create_user("bob", "a long passphrase")
    await accounts.open_org(alice.user_id, "org-a")
    assert not await accounts.open_org(bob.user_id, "org-a")
    assert await accounts.user_orgs(bob.user_id) == []
    await accounts.grant("bob", "org-a")
    assert await accounts.open_org(bob.user_id, "org-a")


async def test_grant_creates_the_org_and_is_idempotent_and_needs_a_real_user(database):
    from agent import accounts

    alice = await accounts.create_user("alice", "a long passphrase")
    await accounts.grant("alice", "org-b")
    await accounts.grant("alice", "org-b")
    assert await accounts.user_orgs(alice.user_id) == ["org-b"]
    with pytest.raises(accounts.AccountError, match="no user"):
        await accounts.grant("nobody", "org-b")


async def test_revoke_takes_away_one_org(database):
    from agent import accounts

    alice = await accounts.create_user("alice", "a long passphrase")
    for org in ("org-a", "org-b"):
        await accounts.grant("alice", org)
    assert await accounts.revoke("alice", "org-a")
    assert await accounts.user_orgs(alice.user_id) == ["org-b"]
    assert not await accounts.revoke("alice", "org-a")
    assert not await accounts.open_org(alice.user_id, "org-a")


async def test_list_users_shows_each_user_with_their_orgs(database):
    from agent import accounts

    assert await accounts.list_users() == []
    await accounts.create_user("bob", "a long passphrase")
    await accounts.create_user("alice", "a long passphrase")
    await accounts.grant("alice", "org-b")
    await accounts.grant("alice", "org-a")
    assert await accounts.list_users() == [("alice", ["org-a", "org-b"]), ("bob", [])]


async def test_ensure_user_creates_an_account_once_and_never_changes_it(database):
    from agent import accounts

    assert await accounts.ensure_user("Admin", "the first passphrase")
    assert not await accounts.ensure_user("admin", "a different passphrase")
    assert await accounts.authenticate("admin", "the first passphrase") is not None
    assert await accounts.authenticate("admin", "a different passphrase") is None


async def test_ensure_user_refuses_a_bad_name(database):
    from agent import accounts

    with pytest.raises(accounts.AccountError, match="1 to 64 characters"):
        await accounts.ensure_user("not a name", "a long passphrase")
    assert await accounts.list_users() == []


async def test_has_access_follows_memberships_and_deleted_accounts(database):
    from agent import accounts

    alice = await accounts.create_user("alice", "a long passphrase")
    assert not await accounts.has_access(alice.user_id, "org-a")
    await accounts.grant("alice", "org-a")
    assert await accounts.has_access(alice.user_id, "org-a")
    assert not await accounts.has_access(alice.user_id, "org-b")
    await accounts.revoke("alice", "org-a")
    assert not await accounts.has_access(alice.user_id, "org-a")
    await accounts.grant("alice", "org-a")
    await accounts.delete_user("alice")
    assert not await accounts.has_access(alice.user_id, "org-a")


async def test_a_password_change_ends_sessions_opened_with_the_old_one(database):
    """Regression: after passwd, a stolen login cookie kept working for up to a day."""
    from agent import accounts

    alice = await accounts.create_user("alice", "a long passphrase")
    await accounts.grant("alice", "org-a")
    logged_in = await accounts.authenticate("alice", "a long passphrase")
    assert logged_in == alice == await accounts.get_user(alice.user_id)
    assert await accounts.has_access(alice.user_id, "org-a", alice.session_key)
    await accounts.set_password("alice", "another long passphrase")
    assert not await accounts.has_access(alice.user_id, "org-a", alice.session_key)
    assert await accounts.has_access(alice.user_id, "org-a")  # scripts: no session to check
    renewed = await accounts.get_user(alice.user_id)
    assert renewed.session_key != alice.session_key
    assert await accounts.has_access(alice.user_id, "org-a", renewed.session_key)
