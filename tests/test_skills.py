"""Unit tests for agent/skills.py — loading and validating skills/*.yaml."""

from unittest.mock import patch

import pytest

PLATFORM_YAML = (
    "name: Test Platform\nid: test\ntone: casual\nmax_length: 500\nhashtag_count: '2-4'\n"
)
CONTENT_TYPE_YAML = "kind: content_type\nid: appeal\nname: Appeal\npurpose: Raise money\nstructure: |\n  1. Story\n  2. Ask\n"


@pytest.fixture
def skills_dir(tmp_path):
    with patch("agent.skills.SKILLS_DIR", tmp_path):
        yield tmp_path


def test_load_skills_splits_platforms_and_content_types(skills_dir):
    from agent.skills import load_skills

    (skills_dir / "platform.yaml").write_text(PLATFORM_YAML)
    (skills_dir / "appeal.yaml").write_text(CONTENT_TYPE_YAML)
    skills = load_skills()
    assert [s["id"] for s in skills.platforms] == ["test"]
    assert [s["id"] for s in skills.content_types] == ["appeal"]
    assert skills.problems == []


def test_load_skills_ignores_content_pillars_file(skills_dir):
    from agent.skills import load_skills

    (skills_dir / "content_pillars.yaml").write_text("pillars:\n  - name: Some Pillar\n")
    skills = load_skills()
    assert skills.platforms == [] and skills.problems == []


@pytest.mark.parametrize(
    "content, expected_problem",
    [
        ("name: No Id\nmax_length: 500\n", "missing required field(s) id"),
        ("id: x\nname: X\nmax_length: lots\n", "max_length must be a whole number"),
        ("kind: video\nid: x\nname: X\n", "kind must be one of"),
        ("kind: content_type\nid: x\nname: X\n", "missing required field(s) purpose, structure"),
        ("id: x\nname: [unclosed\n", "not valid YAML"),
        ("- just\n- a list\n", "expected 'field: value' lines"),
    ],
)
def test_load_skills_reports_and_skips_invalid_files(skills_dir, content, expected_problem):
    from agent.skills import load_skills

    (skills_dir / "good.yaml").write_text(PLATFORM_YAML)
    (skills_dir / "bad.yaml").write_text(content)
    skills = load_skills()
    assert [s["id"] for s in skills.platforms] == ["test"]  # the good file still loads
    assert len(skills.problems) == 1
    assert skills.problems[0].startswith("bad.yaml:")
    assert expected_problem in skills.problems[0]


def test_load_skills_reports_duplicate_ids(skills_dir):
    from agent.skills import load_skills

    (skills_dir / "a.yaml").write_text(PLATFORM_YAML)
    (skills_dir / "b.yaml").write_text(PLATFORM_YAML)
    skills = load_skills()
    assert len(skills.platforms) == 1
    assert "already used" in skills.problems[0]


def test_load_content_pillars_prefers_profile_override(skills_dir):
    from agent.skills import load_content_pillars

    (skills_dir / "content_pillars.yaml").write_text(
        "pillars:\n  - name: Default\n    description: d\n"
    )
    profile = {"content_pillars": [{"name": "Custom", "description": "c"}]}
    assert [p["name"] for p in load_content_pillars(profile)] == ["Custom"]
    assert [p["name"] for p in load_content_pillars({})] == ["Default"]


def test_load_content_pillars_survives_broken_file(skills_dir):
    from agent.skills import load_content_pillars

    (skills_dir / "content_pillars.yaml").write_text("pillars: [unclosed\n")
    assert load_content_pillars({}) == []


def test_load_content_pillars_drops_entries_without_name(skills_dir):
    from agent.skills import load_content_pillars

    profile = {"content_pillars": [{"description": "no name"}, {"name": "Ok"}]}
    assert load_content_pillars(profile) == [{"name": "Ok", "description": ""}]


@pytest.mark.parametrize(
    "value, expected",
    [
        ("3-5", (3, 5)),
        ("2", (2, 2)),
        (0, (0, 0)),
        ("a few", None),
        (None, None),
    ],
)
def test_hashtag_range_parses_common_forms(value, expected):
    from agent.skills import hashtag_range

    assert hashtag_range({"hashtag_count": value}) == expected


def test_shipped_skill_files_are_all_valid():
    """Guards the real skills/ folder: every shipped file must load cleanly."""
    from agent.skills import load_skills

    skills = load_skills()
    assert skills.problems == []
    assert {"linkedin", "instagram", "facebook"} <= {s["id"] for s in skills.platforms}
    assert len(skills.content_types) >= 3


@pytest.mark.parametrize(
    "extra, expected_problem",
    [
        ("format_hints:\n  - Short\n  - Punchy\n", "format_hints must be text"),
        ("tone: 3\n", "tone must be text"),
    ],
)
def test_load_skills_reports_platform_fields_with_the_wrong_type(
    skills_dir, extra, expected_problem
):
    """The prompt builder relies on these types; a YAML list instead of a '|' block is easy to write."""
    from agent.skills import load_skills

    (skills_dir / "bad.yaml").write_text(PLATFORM_YAML + extra)
    skills = load_skills()
    assert skills.platforms == []
    assert expected_problem in skills.problems[0]


@pytest.mark.parametrize(
    "extra, expected_problem",
    [
        ("must_include: one link\n", "must_include must be a list"),
        ("avoid:\n  - guilt\n  - 3\n", "avoid must be a list"),
    ],
)
def test_load_skills_reports_content_type_lists_with_the_wrong_type(
    skills_dir, extra, expected_problem
):
    from agent.skills import load_skills

    (skills_dir / "bad.yaml").write_text(CONTENT_TYPE_YAML + extra)
    skills = load_skills()
    assert skills.content_types == []
    assert expected_problem in skills.problems[0]


def test_load_skills_reports_a_non_text_id_instead_of_crashing(skills_dir):
    """An id like [a, b] is unhashable; the duplicate-id check must not raise on it."""
    from agent.skills import load_skills

    (skills_dir / "bad.yaml").write_text("id: [a, b]\nname: X\nmax_length: 100\n")
    assert "id must be text" in load_skills().problems[0]
