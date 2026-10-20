"""
Unit tests for agent/skills.py: loading and validating skills/*.yaml.

Owns: load_skills (sorting files into platforms, content types and formats;
skipping and reporting a missing or wrongly typed field, bad YAML, a non-UTF-8
or unreadable file, a duplicate id), the content pillars (file, per-org
override, problems), hashtag_range, the shipped skills/ files, and the same
result with or without PyYAML's C parser. Does not own how skills reach the
prompt (test_assistant.py) or the validate.py report (test_validate.py).
"""

from unittest.mock import patch

import pytest

PLATFORM_YAML = (
    "name: Test Platform\nid: test\ntone: casual\nmax_length: 500\nhashtag_count: '2-4'\n"
)
CONTENT_TYPE_YAML = "kind: content_type\nid: appeal\nname: Appeal\npurpose: Raise money\nstructure: |\n  1. Story\n  2. Ask\n"
FORMAT_YAML = "kind: format\nid: reel\nname: Reel\npurpose: Reach\nstructure: Hook\nplatforms:\n  - instagram\n"
STORY_YAML = "kind: format\nid: story\nname: Story\npurpose: Connect\nstructure: Frames\n"
NOT_UTF8 = b"id: caf\xe9\n"  # Latin-1, as a Windows editor may save it
A_FOLDER = None  # stands for a folder named *.yaml instead of a file


def _write(path, content):
    """Write text or bytes to path, or make a folder there for A_FOLDER."""
    if content is A_FOLDER:
        path.mkdir()
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


@pytest.fixture
def skills_dir(tmp_path):
    with patch("agent.skills.SKILLS_DIR", tmp_path):
        yield tmp_path


def test_load_skills_splits_platforms_content_types_and_formats(skills_dir):
    from agent.skills import load_skills

    (skills_dir / "platform.yaml").write_text(PLATFORM_YAML)
    (skills_dir / "appeal.yaml").write_text(CONTENT_TYPE_YAML)
    (skills_dir / "reel.yaml").write_text(FORMAT_YAML)
    skills = load_skills()
    assert [s["id"] for s in skills.platforms] == ["test"]
    assert [s["id"] for s in skills.content_types] == ["appeal"]
    assert [f["id"] for f in skills.formats] == ["reel"]
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
        # A list is unhashable: the kind check itself must not raise on it.
        ("kind: [a]\nid: x\nname: X\n", "kind must be one of"),
        # bool is an int in Python, but true is no length.
        ("id: x\nname: X\nmax_length: true\n", "max_length must be a whole number"),
        ("kind: content_type\nid: x\nname: X\n", "missing required field(s) purpose, structure"),
        ("id: x\nname: [unclosed\n", "not valid YAML"),
        ("- just\n- a list\n", "expected 'field: value' lines"),
        (NOT_UTF8, "not saved as UTF-8"),
        (A_FOLDER, "can't be read"),
        # Regression: ids are matched lowercased, so "LinkedIn" silently skipped its rules.
        ("id: LinkedIn\nname: X\nmax_length: 5\n", "'LinkedIn' must be written in lowercase"),
        ("id: x-twitter\nname: X\nmax_length: 5\n", "'x-twitter' must be written in lowercase"),
        (
            "kind: format\nid: x\nname: X\npurpose: p\nstructure: s\nplatforms: [Instagram]\n",
            "'Instagram' must be written in lowercase",
        ),
    ],
)
def test_load_skills_reports_and_skips_invalid_files(skills_dir, content, expected_problem):
    from agent.skills import load_skills

    (skills_dir / "good.yaml").write_text(PLATFORM_YAML)
    _write(skills_dir / "bad.yaml", content)
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


@pytest.mark.parametrize(
    "text, problem",
    [
        ("pillars: [unclosed\n", "not valid YAML"),
        ("- just a list\n", "expected a 'pillars:' list"),
        ("pillars: Community\n", "expected a 'pillars:' list"),
        ("pillars: 5\n", "expected a 'pillars:' list"),
        ("pillars:\n  - description: no name\n", "every pillar needs a name"),
        (NOT_UTF8, "not saved as UTF-8"),
        (A_FOLDER, "can't be read"),
    ],
)
def test_a_broken_pillars_file_means_no_pillars_and_is_reported(skills_dir, text, problem):
    """No crash for the conversation; a plain-English problem for the UI and validate.py."""
    from agent.skills import load_content_pillars, load_skills

    _write(skills_dir / "content_pillars.yaml", text)
    assert load_content_pillars({}) == []
    problems = load_skills().problems
    assert len(problems) == 1
    assert problems[0].startswith("content_pillars.yaml: ") and problem in problems[0]


def test_a_missing_pillars_file_is_not_a_problem(skills_dir):
    from agent.skills import load_content_pillars, load_skills

    assert load_content_pillars({}) == []
    assert load_skills().problems == []


def test_load_content_pillars_drops_entries_without_name(skills_dir):
    from agent.skills import load_content_pillars

    profile = {"content_pillars": [{"description": "no name"}, {"name": "Ok"}]}
    assert load_content_pillars(profile) == [{"name": "Ok", "description": ""}]


def test_per_org_pillars_that_are_not_a_list_mean_no_pillars(skills_dir):
    from agent.skills import load_content_pillars

    assert load_content_pillars({"content_pillars": 5}) == []


@pytest.mark.parametrize(
    "value, expected",
    [
        ("3-5", (3, 5)),
        ("2", (2, 2)),
        (0, (0, 0)),
        ("a few", None),
        (None, None),
        (True, None),
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
    assert {"linkedin", "instagram", "facebook", "email"} <= {s["id"] for s in skills.platforms}
    assert len(skills.content_types) >= 3
    assert {"feed_post", "carousel", "story", "reel"} <= {f["id"] for f in skills.formats}


@pytest.mark.parametrize(
    "content, expected_problem",
    [
        # The prompt builder relies on these types; a YAML list instead of a '|' block is easy
        # to write.
        (PLATFORM_YAML + "format_hints:\n  - Short\n  - Punchy\n", "format_hints must be text"),
        (PLATFORM_YAML + "tone: 3\n", "tone must be text"),
        (CONTENT_TYPE_YAML + "must_include: one link\n", "must_include must be a list"),
        (CONTENT_TYPE_YAML + "avoid:\n  - guilt\n  - 3\n", "avoid must be a list"),
        (STORY_YAML + "has_caption: no-thanks\n", "has_caption must be true or false"),
        (STORY_YAML + "needs_visual: maybe\n", "needs_visual must be true or false"),
        (STORY_YAML + "platforms: instagram\n", "platforms must be a list"),
        # Rendered as "on all platforms" but matching none: leaving it out means all.
        (STORY_YAML + "platforms: []\n", "list the platforms under platforms, or leave it out"),
        # An id like [a, b] is unhashable; the duplicate-id check must not raise on it.
        ("id: [a, b]\nname: X\nmax_length: 100\n", "id must be text"),
    ],
)
def test_load_skills_reports_and_skips_a_field_with_the_wrong_type(
    skills_dir, content, expected_problem
):
    from agent.skills import load_skills

    (skills_dir / "bad.yaml").write_text(content)
    skills = load_skills()
    assert skills.platforms == [] and skills.content_types == [] and skills.formats == []
    assert len(skills.problems) == 1
    assert expected_problem in skills.problems[0]


def test_skills_read_the_same_with_or_without_pyyamls_c_parser():
    """The C parser is only faster; a PyYAML built without it must load identical skills."""
    import yaml

    from agent import skills

    with_c = skills.load_skills()
    with patch("agent.skills._YAML_LOADER", yaml.SafeLoader):
        without_c = skills.load_skills()
    assert with_c == without_c and with_c.platforms
    assert skills._YAML_LOADER in (getattr(yaml, "CSafeLoader", None), yaml.SafeLoader)
