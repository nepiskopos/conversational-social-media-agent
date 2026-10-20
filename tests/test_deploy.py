"""
Regression tests for deployment settings that protect secrets and the server.

Owns: checks on .gitignore, .dockerignore, .chainlit/config.toml,
docker-compose.yml, the Dockerfile, docker/postgres-init.sh, .env.example and
the CI workflow's time budget and stack cleanup.
These files are not code, so nothing else would notice a change that ships a
key in the Docker image, reopens file uploads, publishes the database port or
hands the app a superuser password. Also checks that the installed
LiteLLM can make a tool call, which every other test mocks away.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from dotenv import dotenv_values

# Bound before conftest's autouse fixture replaces litellm.acompletion.
from litellm import acompletion as real_acompletion

ROOT = Path(__file__).parent.parent
# Secrets the app container must not receive: a compromised app could log in with them as
# the superuser, the schema owner or the default account.
APP_HIDDEN_SECRETS: set[str] = {
    "MACKIE_MIGRATION_URL",
    "MACKIE_DEFAULT_PASSWORD",
    "POSTGRES_PASSWORD",
    "MACKIE_OWNER_PASSWORD",
    "MACKIE_APP_PASSWORD",
}
COMPOSE_PASSWORDS: list[str] = ["POSTGRES_PASSWORD", "MACKIE_OWNER_PASSWORD", "MACKIE_APP_PASSWORD"]
ONE_DAY_S: int = 86_400


def _patterns(file_name: str) -> set[str]:
    """The non-comment lines of an ignore file."""
    lines = (ROOT / file_name).read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_secrets_and_user_data_are_kept_out_of_git():
    patterns = _patterns(".gitignore")
    assert {".env", ".env.*", "!.env.example", "data/*"} <= patterns


def test_secrets_user_data_and_bytecode_are_kept_out_of_the_docker_image():
    """.dockerignore patterns match only at the top level unless they start with **/."""
    patterns = _patterns(".dockerignore")
    assert {"**/.env", "**/.env.*", "data/", "**/__pycache__/", "**/*.py[cod]"} <= patterns


def _chainlit_config() -> dict:
    return tomllib.loads((ROOT / ".chainlit" / "config.toml").read_text())


@pytest.mark.parametrize(
    "path, allowed",
    [
        # Mackie never reads uploads; accepting them (500 MB each by default) only fills the disk.
        (("features", "spontaneous_file_upload", "enabled"), lambda value: value is False),
        # Replies contain scraped text; raw HTML rendering would let a web page inject markup.
        (("features", "unsafe_allow_html"), lambda value: value is False),
        # Same origin only; "*" would let any site drive a logged-in session.
        (("project", "allow_origins"), lambda value: value == []),
        # Logins expire within a day.
        (("project", "user_session_timeout"), lambda value: 0 < value <= ONE_DAY_S),
        # An edit reruns the turn, but the stored conversation would keep the original.
        (("features", "edit_message"), lambda value: value is False),
    ],
    ids=["no-uploads", "no-raw-html", "same-origin", "logins-expire", "no-edits"],
)
def test_chainlit_settings_protect_the_server_and_sessions(path, allowed):
    value = _chainlit_config()
    for key in path:
        value = value[key]
    assert allowed(value), (path, value)


async def test_installed_litellm_accepts_tool_calls():
    """
    LiteLLM 1.106.0 and 1.106.1 import mcp>=2.2 on every call with tools, but Chainlit pins
    mcp<2, so every chat turn failed with any provider. mock_response stays offline.
    """
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    response = await real_acompletion(
        model="openai/any",
        api_key="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=[tool],
        mock_response="ok",
    )
    assert response.choices[0].message.content == "ok"


def _compose_services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def test_compose_publishes_no_database_port():
    """Only the services on compose's network may reach PostgreSQL."""
    postgres = _compose_services()["postgres"]
    assert "ports" not in postgres and "expose" not in postgres


def test_compose_keeps_admin_secrets_out_of_the_app_container():
    """env_file passes all of .env, so each secret the app mustn't hold is blanked over it."""
    environment = _compose_services()["mackie"]["environment"]
    assert {name: environment.get(name) for name in APP_HIDDEN_SECRETS} == dict.fromkeys(
        APP_HIDDEN_SECRETS, ""
    )
    assert environment["MACKIE_DATABASE_URL"].startswith("postgresql://mackie_app:")


def test_compose_serves_the_app_only_to_this_machine_and_mounts_skills_read_only():
    """Plain HTTP stays off the network (a TLS proxy serves others); the app can't edit skills."""
    mackie = _compose_services()["mackie"]
    assert mackie["ports"] == ["127.0.0.1:8000:8000"]
    assert "./skills:/app/skills:ro" in mackie["volumes"]


def test_compose_refuses_to_start_without_any_password_or_secret():
    """
    Every password and secret compose defines is ${NAME:?...}: an unset one would otherwise
    start the database or the default account with an empty password, or a web UI whose
    every login fails. Blanked ones are not defined.
    """
    defined = {
        (service_name, name): value
        for service_name, service in _compose_services().items()
        for name, value in (service.get("environment") or {}).items()
        if name.endswith(("_PASSWORD", "_SECRET")) and value != ""
    }
    expected = {("postgres", name) for name in COMPOSE_PASSWORDS} | {
        ("migrate", "MACKIE_DEFAULT_PASSWORD"),
        ("mackie", "CHAINLIT_AUTH_SECRET"),
    }
    assert set(defined) == expected
    for (_, name), value in defined.items():
        assert re.fullmatch(rf"\$\{{{name}:\?[^}}]+\}}", value), (name, value)


def test_docker_image_runs_as_an_unprivileged_user():
    """USER comes after every step that needs root, so the app process itself never is."""
    steps = [
        line.split(maxsplit=1)
        for line in (ROOT / "Dockerfile").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    instructions = [step[0].upper() for step in steps]
    last_user = len(instructions) - 1 - instructions[::-1].index("USER")
    assert steps[last_user][1].strip() not in {"root", "0", "0:0"}
    assert not {"RUN", "COPY", "ADD"} & set(instructions[last_user:])


def test_postgres_init_creates_login_roles_without_admin_rights():
    script = (ROOT / "docker" / "postgres-init.sh").read_text()
    for role in ("mackie_owner", "mackie_app"):
        created = re.search(rf"CREATE ROLE {role}\b([^;]*);", script)
        assert created, role
        options = created.group(1).upper()
        assert re.search(r"\bLOGIN\b", options), role
        for right in ("SUPERUSER", "CREATEROLE", "CREATEDB", "BYPASSRLS"):
            assert not re.search(rf"\b{right}\b", options), (role, right)
    assert "REVOKE CREATE ON SCHEMA public FROM PUBLIC;" in script


def test_env_example_documents_every_database_and_login_secret_without_a_real_value():
    settings = dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    passwords = ["MACKIE_DEFAULT_PASSWORD", "CHAINLIT_AUTH_SECRET", *COMPOSE_PASSWORDS]
    for name in passwords:
        assert settings[name].strip() == "", name
    for name in ("MACKIE_DATABASE_URL", "MACKIE_MIGRATION_URL"):
        assert ":CHANGE_ME@" in settings[name], name


def test_env_example_leaves_every_key_blank():
    """Any value makes a provider count as configured: a placeholder key became the main model."""
    keys = {
        name: value
        for name, value in dotenv_values(ROOT / ".env.example").items()
        if name.endswith("_API_KEY")
    }
    assert "TAVILY_API_KEY" in keys and "ANTHROPIC_API_KEY" in keys
    assert {name: value for name, value in keys.items() if value} == {}


def test_env_example_puts_comments_on_their_own_lines():
    """
    Compose's env_file reads `NAME=  # note` as the value "# note", unlike python-dotenv, so
    an inline comment works in a local run and breaks the same setting in Docker.
    """
    inline = [
        line
        for line in (ROOT / ".env.example").read_text().splitlines()
        if not line.lstrip().startswith("#") and "=" in line and "#" in line
    ]
    assert inline == []


def test_ci_gives_the_suite_room_and_cleans_up_its_docker_stack():
    """
    Shared runners vary in speed, so CI raises the time budget; and the deployment job must
    never share a project (and its database volume) with a real stack, and must remove its
    own however the job ends.
    """
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    tests = next(step for step in ci["jobs"]["test"]["steps"] if step.get("name") == "Run tests")
    assert float(tests["env"]["MACKIE_TEST_BUDGET_S"]) > 30
    deploy = ci["jobs"]["deploy"]
    assert deploy["env"]["COMPOSE_PROJECT_NAME"] == "mackie-ci"
    # A hung browser or stack stops within minutes, not GitHub's default six hours.
    assert all(0 < job["timeout-minutes"] <= 30 for job in ci["jobs"].values())
    # The jobs only read the repository, so a compromised step can't push or publish.
    assert ci["permissions"] == {"contents": "read"}
    last = deploy["steps"][-1]
    assert last["if"] == "always()" and "docker compose down --volumes" in last["run"]
