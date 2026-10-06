"""Unit tests for scripts/validate.py — the plain-English check of hand-edited files."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

VALIDATE_PATH = Path(__file__).parent.parent / "scripts" / "validate.py"
PLATFORM_YAML = "name: Test\nid: test\nmax_length: 500\n"


@pytest.fixture(scope="module")
def validate_module():
    """Import scripts/validate.py as a module."""
    spec = importlib.util.spec_from_file_location("mackie_validate", VALIDATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules.pop("mackie_validate", None)
    return module


@pytest.fixture
def skills_dir(tmp_path):
    folder = tmp_path / "skills"
    folder.mkdir()
    (folder / "test.yaml").write_text(PLATFORM_YAML)
    with patch("agent.skills.SKILLS_DIR", folder):
        yield folder


def test_validate_passes_valid_files(validate_module, tmp_data_dir, skills_dir, capsys):
    (tmp_data_dir / "org").mkdir()
    (tmp_data_dir / "org" / "profile.json").write_text('{"name": "Org"}')
    assert validate_module.main() == 0
    out = capsys.readouterr().out
    assert "Platforms:     test" in out
    assert "All skill files and org profiles are valid." in out


def test_validate_reports_every_broken_file(validate_module, tmp_data_dir, skills_dir, capsys):
    (skills_dir / "broken.yaml").write_text("id: x\nname: X\nmax_length: lots\n")
    (tmp_data_dir / "org").mkdir()
    (tmp_data_dir / "org" / "profile.json").write_text('{"name": "Org",}')
    assert validate_module.main() == 1
    out = capsys.readouterr().out
    assert "2 problem(s):" in out
    assert "broken.yaml: max_length must be a whole number" in out
    assert "formatting error at line 1" in out


def test_validate_reports_a_folder_mackie_never_opens(validate_module, tmp_data_dir, skills_dir):
    """A hand-made folder like "My Org" is not an org id, so its profile would silently be ignored."""
    (tmp_data_dir / "My Org").mkdir()
    (tmp_data_dir / "My Org" / "profile.json").write_text("{}")
    problems = validate_module.profile_problems()
    assert len(problems) == 1
    assert "is not an org id" in problems[0]


def test_validate_changes_nothing_on_disk(validate_module, tmp_data_dir, skills_dir):
    (tmp_data_dir / "org").mkdir()
    profile = tmp_data_dir / "org" / "profile.json"
    profile.write_text('{"name": "Org"}')
    validate_module.main()
    assert sorted(p.name for p in (tmp_data_dir / "org").iterdir()) == ["profile.json"]
    assert profile.read_text() == '{"name": "Org"}'
