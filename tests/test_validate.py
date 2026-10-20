"""
Unit tests for scripts/validate.py: the plain-English check of hand-edited files.

Owns: the report for valid files (and that the check changes nothing), every
kind of broken skill, pillars or profile file in one report, and a profile
file that can't be read. Does not own the individual checks (test_skills.py,
test_memory.py for profile_problem).
"""

import logging
from unittest.mock import patch

import pytest

from tests.conftest import load_script

PLATFORM_YAML = "name: Test\nid: test\nmax_length: 500\n"
GOOD_PROFILE = (
    '{"org_id": "org", "name": "Org", "post_ledger": [{"id": "post_001", "title": "Hello"}]}'
)


@pytest.fixture(scope="module")
def validate_module():
    return load_script("validate")


@pytest.fixture
def skills_dir(tmp_path):
    folder = tmp_path / "skills"
    folder.mkdir()
    (folder / "test.yaml").write_text(PLATFORM_YAML)
    with patch("agent.skills.SKILLS_DIR", folder):
        yield folder


def test_validate_passes_valid_files_and_changes_nothing(
    validate_module, skills_dir, tmp_path, capsys
):
    """Nothing on disk changes, and without the database fixture any database use would fail."""
    profile = tmp_path / "org.json"
    profile.write_text(GOOD_PROFILE)
    before = sorted(tmp_path.rglob("*"))
    assert validate_module.main([str(profile)]) == 0
    out = capsys.readouterr().out
    assert "Platforms:     test" in out
    assert "All files are valid." in out
    assert sorted(tmp_path.rglob("*")) == before
    assert profile.read_text() == GOOD_PROFILE
    assert (skills_dir / "test.yaml").read_text() == PLATFORM_YAML


def test_validate_help_prints_usage_instead_of_checking_a_file_named_help(validate_module, capsys):
    with pytest.raises(SystemExit) as stopped:
        validate_module.main(["--help"])
    assert stopped.value.code == 0
    assert "usage: " in capsys.readouterr().out


def test_validate_reports_every_broken_file(validate_module, skills_dir, tmp_path, capsys, caplog):
    (skills_dir / "broken.yaml").write_text("id: x\nname: X\nmax_length: lots\n")
    (skills_dir / "content_pillars.yaml").write_text("pillars:\n  - description: no name\n")
    profiles = {
        "comma.json": '{"name": "Org",}',
        "list.json": "[]",
        "status.json": '{"post_ledger": [{"status": "maybe"}]}',
        # Regression: passed here, then the import refused it for having no org.
        "no-org.json": '{"name": "Org"}',
    }
    for name, text in profiles.items():
        (tmp_path / name).write_text(text)
    assert validate_module.main([str(tmp_path / name) for name in profiles]) == 1
    out = capsys.readouterr().out
    assert "6 problem(s):" in out
    # Regression: the skills module's log repeated each skill problem; logging is back on after.
    assert caplog.records == []
    assert not logging.root.manager.disable
    assert "broken.yaml: max_length must be a whole number" in out
    assert "content_pillars.yaml: every pillar needs a name" in out
    assert "comma.json has a formatting error at line 1" in out
    assert "list.json must be a JSON object" in out
    assert "status.json: post 1 has status 'maybe'" in out
    assert "no-org.json: '' is not an org id" in out


def test_validate_reports_a_profile_file_it_cant_read(validate_module, skills_dir, tmp_path):
    missing = tmp_path / "missing.json"
    problems = validate_module.profile_problems([missing, tmp_path])
    assert problems[0] == f"{missing}: can't be read (No such file or directory)"
    assert problems[1].startswith(f"{tmp_path}: can't be read")
    assert len(problems) == 2
