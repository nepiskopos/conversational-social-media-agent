"""
Tests for the admin scripts: migrate.py (schema and grants), users.py (accounts
and access), export_profile.py and import_profiles.py (editing an org's memory
as JSON, and moving the old data/ folders into the database). The account and
storage rules themselves are owned by test_accounts.py and test_memory.py.
"""

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import load_script


@pytest.fixture(scope="module")
def migrate():
    return load_script("migrate")


@pytest.fixture(scope="module")
def users():
    return load_script("users")


@pytest.fixture(scope="module")
def export_profile():
    return load_script("export_profile")


@pytest.fixture(scope="module")
def import_profiles():
    return load_script("import_profiles")


# --- migrate.py --------------------------------------------------------------------


def test_migrate_help_prints_usage_instead_of_migrating(migrate, capsys):
    with pytest.raises(SystemExit) as stopped:
        migrate.parse_args(["--help"])
    assert stopped.value.code == 0
    assert "usage: " in capsys.readouterr().out
    assert migrate.parse_args([]) is not None


async def test_migrate_needs_the_owner_connection_string(migrate, capsys):
    assert await migrate.run() == 1
    assert "MACKIE_MIGRATION_URL is not set" in capsys.readouterr().out


async def test_migrate_refuses_one_role_for_app_and_owner(migrate, database, monkeypatch, capsys):
    """An app role that owns the schema could switch row-level security off."""
    monkeypatch.setenv("MACKIE_MIGRATION_URL", database["owner"])
    monkeypatch.setenv("MACKIE_DATABASE_URL", database["owner"])
    assert await migrate.run() == 1
    assert "use the same role" in capsys.readouterr().out


async def test_migrate_on_an_up_to_date_schema_applies_nothing(
    migrate, database, monkeypatch, capsys
):
    monkeypatch.setenv("MACKIE_MIGRATION_URL", database["owner"])
    assert await migrate.run() == 0
    out = capsys.readouterr().out
    assert "up to date" in out and "mackie_test_app can read and write rows" in out


async def test_migrate_reports_a_connection_failure(migrate, database, monkeypatch, capsys):
    monkeypatch.setenv(
        "MACKIE_MIGRATION_URL", database["owner"].replace("password=", "password=wrong")
    )
    assert await migrate.run() == 1
    out = capsys.readouterr().out
    assert "Migration failed" in out
    assert "wrong" not in out  # the password from the connection string is never shown


async def test_migrate_creates_the_configured_default_account_once(
    migrate, database, monkeypatch, capsys
):
    from agent import accounts

    monkeypatch.setenv("MACKIE_MIGRATION_URL", database["owner"])
    monkeypatch.setenv("MACKIE_DEFAULT_USERNAME", "Owner@Food.example")
    monkeypatch.setenv("MACKIE_DEFAULT_PASSWORD", "configured passphrase")
    assert await migrate.run() == 0
    out = capsys.readouterr().out
    assert "Created the default account owner@food.example" in out
    assert "configured passphrase" not in out
    assert await accounts.authenticate("owner@food.example", "configured passphrase")

    await accounts.set_password("owner@food.example", "changed by the admin")
    assert await migrate.run() == 0  # a redeploy
    assert "The default account owner@food.example exists" in capsys.readouterr().out
    assert await accounts.authenticate("owner@food.example", "changed by the admin")


async def test_migrate_without_a_default_password_makes_one_and_shows_it_once(
    migrate, database, monkeypatch, capsys
):
    from agent import accounts

    monkeypatch.setenv("MACKIE_MIGRATION_URL", database["owner"])
    assert await migrate.run() == 0
    out = capsys.readouterr().out
    password = out.split("with the password ")[1].split(" ")[0]
    assert len(password) >= accounts.MIN_PASSWORD_CHARS
    assert await accounts.authenticate(migrate.DEFAULT_USERNAME, password)
    assert await migrate.run() == 0
    assert password not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("MACKIE_DEFAULT_PASSWORD", "short", "A password needs at least 12 characters"),
        ("MACKIE_DEFAULT_USERNAME", "bad name!", "A username is 1 to 64 characters"),
    ],
)
async def test_migrate_refuses_an_invalid_default_account_after_the_schema_is_ready(
    migrate, database, monkeypatch, capsys, variable, value, message
):
    """The schema step's success is still reported, so the admin knows only the account failed."""
    from agent import accounts

    monkeypatch.setenv("MACKIE_MIGRATION_URL", database["owner"])
    monkeypatch.setenv(variable, value)
    assert await migrate.run() == 1
    out = capsys.readouterr().out
    assert "The schema is up to date." in out
    assert f"The schema is ready, but the default account wasn't created: {message}" in out
    assert await accounts.list_users() == []


# --- users.py ----------------------------------------------------------------------


async def _users(users, argv: list[str], stdin: str = "") -> int:
    with patch("sys.stdin", io.StringIO(stdin)):
        return await users.main(argv)


async def test_users_add_grant_list_revoke_remove(users, database, capsys):
    from agent import accounts

    assert await _users(users, ["add", "Alice", "--password-stdin"], "a long passphrase\n") == 0
    assert await _users(users, ["grant", "alice", "BRCA Strong"]) == 0
    assert await _users(users, ["list"]) == 0
    assert "alice: brca-strong" in capsys.readouterr().out
    assert await accounts.authenticate("alice", "a long passphrase") is not None

    assert await _users(users, ["revoke", "alice", "brca-strong"]) == 0
    assert await _users(users, ["remove", "alice"]) == 0
    assert await _users(users, ["list"]) == 0
    assert "No users yet" in capsys.readouterr().out


async def test_users_passwd_changes_the_password(users, database):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    assert await _users(users, ["passwd", "alice", "--password-stdin"], "a new passphrase!\n") == 0
    assert await accounts.authenticate("alice", "a new passphrase!") is not None


@pytest.mark.parametrize(
    ("argv", "stdin", "message"),
    [
        (["add", "bob", "--password-stdin"], "short\n", "at least 12"),
        (["passwd", "nobody", "--password-stdin"], "a long passphrase\n", "no user nobody"),
        (["remove", "nobody"], "", "no user nobody"),
        (["grant", "nobody", "org-a"], "", "no user nobody"),
        (["grant", "alice", "!!!"], "", "not an organization name"),
        (["revoke", "alice", "org-a"], "", "can't open org-a anyway"),
    ],
)
async def test_users_explains_what_went_wrong(users, database, capsys, argv, stdin, message):
    from agent import accounts

    await accounts.create_user("alice", "a long passphrase")
    assert await _users(users, argv, stdin) == 1
    assert message in capsys.readouterr().out


def test_users_asks_for_the_password_twice_without_echo(users):
    from agent.accounts import AccountError

    with patch("getpass.getpass", side_effect=["a long passphrase", "a long passphrase"]):
        assert users.read_password(from_stdin=False) == "a long passphrase"
    with (
        patch("getpass.getpass", side_effect=["a long passphrase", "a typo passphrase"]),
        pytest.raises(AccountError, match="don't match"),
    ):
        users.read_password(from_stdin=False)


async def test_users_reports_a_missing_database(users, capsys):
    """get_pool fails as it does on a server where MACKIE_DATABASE_URL was never set."""
    from agent.db import DatabaseError

    with patch("agent.db.get_pool", side_effect=DatabaseError("not set")):
        assert await _users(users, ["list"]) == 1
    assert "not set" in capsys.readouterr().out


# --- export_profile.py and import_profiles.py ---------------------------------------


async def test_export_then_import_round_trips_an_edited_profile(
    export_profile, import_profiles, database, tmp_path, capsys
):
    from agent import memory as mem

    await mem.fill_org_details("brca-strong", "BRCA Strong", "https://brcastrong.org")
    await mem.add_post("brca-strong", {"platform": "linkedin", "content": "Draft"})
    await mem.save_preference("brca-strong", "voice", "warm", quote="be warmer")
    exported = tmp_path / "brca.json"
    assert await export_profile.export("BRCA Strong", exported) == 0

    edited = json.loads(exported.read_text())
    edited["preferences"]["voice"] = "warm and direct"
    edited["post_ledger"].append({"platform": "instagram", "content": "Typed by hand"})
    exported.write_text(json.dumps(edited))
    assert await import_profiles.import_files([exported], check_only=False) == 0
    assert "Imported" in capsys.readouterr().out

    profile = await mem.load_profile("brca-strong")
    assert profile["preferences"] == {"voice": "warm and direct"}
    assert profile["preference_notes"]["voice"]["quote"] == "be warmer"
    assert [post["id"] for post in profile["post_ledger"]] == ["post_001", "post_002"]


async def test_export_writes_to_stdout_and_refuses_a_non_name_or_an_unknown_org(
    export_profile, database, capsys
):
    from agent import memory as mem

    await mem.fill_org_details("new-org", "New Org")
    assert await export_profile.export("New Org", None) == 0
    assert json.loads(capsys.readouterr().out)["org_id"] == "new-org"
    assert await export_profile.export("!!!", None) == 1
    assert "not an organization name" in capsys.readouterr().err
    # A typo would export an empty profile, and importing that would create a new org.
    assert await export_profile.export("New Orgg", None) == 1
    assert "nothing is stored for new-orgg" in capsys.readouterr().err


async def test_import_moves_the_old_data_folders(import_profiles, database, tmp_path, capsys):
    """data/{org}/profile.json files have no org_id of their own: the folder names the org."""
    from agent import memory as mem

    folder = tmp_path / "data" / "doctors-without-borders"
    folder.mkdir(parents=True)
    old = {
        "name": "Doctors Without Borders",
        "preferences": {"voice": "plain"},
        "post_ledger": [
            {"id": "post_003", "platform": "linkedin", "content": "x", "status": "draft"},
            {"id": "post_004", "platform": "facebook", "content": "y", "status": "suggested"},
        ],
    }
    (folder / "profile.json").write_text(json.dumps(old))
    assert await import_profiles.import_files([folder / "profile.json"], check_only=False) == 0
    assert "doctors-without-borders (2 posts)" in capsys.readouterr().out
    profile = await mem.load_profile("doctors-without-borders")
    assert profile["name"] == "Doctors Without Borders"
    assert [p["id"] for p in profile["post_ledger"]] == ["post_003", "post_004"]
    assert await mem.add_post("doctors-without-borders", {"platform": "x", "content": "z"}) == (
        "post_005"
    )


async def test_import_check_changes_nothing(import_profiles, database, tmp_path, capsys):
    from agent import memory as mem

    path = tmp_path / "org.json"
    path.write_text(json.dumps({"org_id": "org-a", "name": "Org A"}))
    assert await import_profiles.import_files([path], check_only=True) == 0
    assert "Valid" in capsys.readouterr().out
    assert (await mem.load_profile("org-a"))["name"] == ""


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "formatting error"),
        (json.dumps({"org_id": "org-a", "post_ledger": [{"status": "live"}]}), "status 'live'"),
        (json.dumps({"org_id": "../org-a"}), "is not an org id"),
        (json.dumps({"name": "No id and not in a data folder"}), "is not an org id"),
    ],
)
@pytest.mark.parametrize("check_only", [True, False], ids=["check", "import"])
async def test_import_refuses_a_broken_file_and_keeps_what_was_stored(
    import_profiles, database, tmp_path, capsys, content, message, check_only
):
    """
    --check refuses exactly what the import would, so a file it passes imports.

    Which profiles are refused, and why, is owned by test_memory.py.
    """
    from agent import memory as mem

    await mem.fill_org_details("org-a", "Org A")
    path = tmp_path / "org.json"
    path.write_text(content)
    assert await import_profiles.import_files([path], check_only=check_only) == 1
    out = capsys.readouterr().out
    assert message in out and f"Not imported: {path}" in out
    assert (await mem.load_profile("org-a"))["name"] == "Org A"


async def test_a_file_the_database_refuses_doesnt_stop_the_batch(
    import_profiles, database, tmp_path, capsys
):
    import psycopg

    from agent import memory as mem

    bad, good = tmp_path / "bad.json", tmp_path / "good.json"
    bad.write_text(json.dumps({"org_id": "org-a", "name": "Org A"}))
    good.write_text(json.dumps({"org_id": "org-b", "name": "Org B"}))
    real_save = mem.save_profile

    async def refuse_the_first(org_id, profile):
        if org_id == "org-a":
            raise psycopg.errors.DataError("value out of range")
        await real_save(org_id, profile)

    with patch("agent.memory.save_profile", refuse_the_first):
        assert await import_profiles.import_files([bad, good], check_only=False) == 1
    out = capsys.readouterr().out
    assert f"Not imported: {bad}: the database refused it" in out
    assert f"Imported: {good}" in out
    assert (await mem.load_profile("org-b"))["name"] == "Org B"
    assert (await mem.load_profile("org-a"))["name"] == ""


async def test_import_reports_a_missing_file_and_goes_on(
    import_profiles, database, tmp_path, capsys
):
    good = tmp_path / "good.json"
    missing = tmp_path / "missing.json"
    good.write_text(json.dumps({"org_id": "org-a"}))
    assert await import_profiles.import_files([missing, good], False) == 1
    out = capsys.readouterr().out
    refused, imported = out.splitlines()
    assert refused.startswith("Not imported") and str(missing) in refused
    assert imported.startswith(f"Imported: {good}")
    assert database["cleaner"].execute("SELECT org_id FROM orgs").fetchall() == [("org-a",)]


def test_entry_points_parse_their_command_lines(export_profile, import_profiles, users):
    with patch.object(export_profile, "export", return_value=0) as export:
        export_profile.main(["BRCA Strong", "-o", "out.json"])
    assert export.call_args.args == ("BRCA Strong", Path("out.json"))
    with patch.object(import_profiles, "import_files", return_value=0) as run:
        import_profiles.main(["--check", "a.json", "b.json"])
    assert run.call_args.args == ([Path("a.json"), Path("b.json")], True)
    args = users.parse_args(["grant", "alice", "BRCA Strong"])
    assert (args.command, args.username, args.org) == ("grant", "alice", "BRCA Strong")
    with pytest.raises(SystemExit):
        users.parse_args([])
