"""
Regression tests for deployment settings that protect secrets and the server.

Owns: checks on .gitignore, .dockerignore and .chainlit/config.toml. These
files are not code, so nothing else would notice a change that ships a key
in the Docker image or reopens file uploads.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _patterns(file_name: str) -> set[str]:
    lines = (ROOT / file_name).read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_secrets_and_user_data_are_kept_out_of_git():
    patterns = _patterns(".gitignore")
    assert {".env", ".env.*", "!.env.example", "data/*"} <= patterns


def test_secrets_and_user_data_are_kept_out_of_the_docker_image():
    """.dockerignore patterns match only at the top level unless they start with **/."""
    patterns = _patterns(".dockerignore")
    assert {"**/.env", "**/.env.*", "data/"} <= patterns


def test_bytecode_caches_in_subfolders_are_kept_out_of_the_docker_image():
    patterns = _patterns(".dockerignore")
    assert {"**/__pycache__/", "**/*.py[cod]"} <= patterns


def test_chainlit_file_uploads_are_off():
    """Mackie never reads uploads; accepting them (500 MB each by default) only fills the disk."""
    config = tomllib.loads((ROOT / ".chainlit" / "config.toml").read_text())
    assert config["features"]["spontaneous_file_upload"]["enabled"] is False


def test_chainlit_does_not_render_raw_html():
    """Replies contain scraped text; raw HTML rendering would let a web page inject markup."""
    config = tomllib.loads((ROOT / ".chainlit" / "config.toml").read_text())
    assert config["features"]["unsafe_allow_html"] is False
