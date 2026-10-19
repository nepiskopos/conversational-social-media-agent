"""
Unit tests for agent/memory.py: profiles (backups, hand-editing errors, the org
lock), the post ledger and its search, post results, the agenda, preferences,
voice examples and org ids.
"""

import json

import pytest


def test_slugify_converts_name_to_id():
    from agent.memory import slugify

    assert slugify("BRCA Strong") == "brca-strong"
    assert slugify("Myra's Kids") == "myra-s-kids"


def test_slugify_collapses_special_chars():
    from agent.memory import slugify

    assert slugify("Vets & Drones") == "vets-drones"
    assert slugify("--Leading Dashes--") == "leading-dashes"


def test_slugify_handles_unicode():
    from agent.memory import slugify

    result = slugify("Café Ñoño")
    # Non-ASCII chars are replaced by hyphens; result must be ASCII and non-empty.
    assert result.isascii()
    assert result


def test_slugify_gives_non_latin_names_distinct_stable_ids():
    from agent.memory import slugify

    greek, japanese = slugify("Ελληνική Ένωση"), slugify("日本")
    assert greek.startswith("org-") and japanese.startswith("org-")
    assert greek != japanese
    assert slugify("Ελληνική Ένωση") == greek
    assert slugify("   ") == ""


def test_load_profile_returns_defaults_for_new_org(tmp_data_dir):
    from agent.memory import load_profile

    profile = load_profile("new-org")
    assert profile["org_id"] == "new-org"
    assert profile["name"] == ""
    assert profile["post_ledger"] == []
    assert profile["preferences"] == {}
    assert profile["research"] is None


def test_save_and_load_roundtrip_creates_the_org_folder(tmp_data_dir):
    from agent.memory import load_profile, save_profile

    profile = load_profile("test-org")
    profile["name"] = "Test Org"
    profile["website"] = "https://testorg.org"
    save_profile("test-org", profile)
    assert (tmp_data_dir / "test-org" / "profile.json").exists()
    loaded = load_profile("test-org")
    assert loaded["name"] == "Test Org"
    assert loaded["website"] == "https://testorg.org"


def test_add_post_assigns_sequential_ids(tmp_data_dir):
    from agent.memory import add_post

    id1 = add_post("org", {"platform": "linkedin", "content": "Post 1"})
    id2 = add_post("org", {"platform": "instagram", "content": "Post 2"})
    id3 = add_post("org", {"platform": "facebook", "content": "Post 3"})
    assert id1 == "post_001"
    assert id2 == "post_002"
    assert id3 == "post_003"


def test_add_post_sets_default_status(tmp_data_dir):
    from agent.memory import add_post, get_ledger

    add_post("org", {"platform": "linkedin", "content": "Hello"})
    assert get_ledger("org")[0]["status"] == "suggested"


def test_add_post_preserves_explicit_status(tmp_data_dir):
    from agent.memory import add_post, get_ledger

    add_post("org", {"platform": "linkedin", "content": "Hello", "status": "draft"})
    assert get_ledger("org")[0]["status"] == "draft"


def test_add_post_sets_created_at(tmp_data_dir):
    from agent.memory import add_post, get_ledger

    add_post("org", {"platform": "linkedin", "content": "Hello"})
    assert "created_at" in get_ledger("org")[0]


def test_update_post_status_success(tmp_data_dir):
    from agent.memory import add_post, get_ledger, update_post_status

    post_id = add_post("org", {"platform": "linkedin", "content": "Hello"})
    result = update_post_status("org", post_id, "approved")
    assert result is True
    assert get_ledger("org")[0]["status"] == "approved"


def test_update_post_status_missing_post_returns_false(tmp_data_dir):
    from agent.memory import update_post_status

    result = update_post_status("org", "post_999", "approved")
    assert result is False


def test_save_preference_persists(tmp_data_dir):
    from agent.memory import load_profile, save_preference

    save_preference("org", "voice", "warm and grassroots")
    assert load_profile("org")["preferences"]["voice"] == "warm and grassroots"


def test_save_preference_overwrites_existing(tmp_data_dir):
    from agent.memory import load_profile, save_preference

    save_preference("org", "voice", "corporate")
    save_preference("org", "voice", "grassroots")
    assert load_profile("org")["preferences"]["voice"] == "grassroots"


def test_get_ledger_empty_for_new_org(tmp_data_dir):
    from agent.memory import get_ledger

    assert get_ledger("new-org") == []


def test_trace_path_returns_correct_path(tmp_data_dir):
    from agent.memory import trace_path

    path = trace_path("my-org")
    assert path.parent == tmp_data_dir / "my-org"
    assert path.name == "trace.jsonl"


def test_add_post_does_not_reuse_id_after_manual_deletion(tmp_data_dir):
    from agent.memory import add_post, load_profile, save_profile

    for i in range(3):
        add_post("org", {"platform": "linkedin", "content": f"post {i}"})
    # A non-engineer deletes post_002 by hand in profile.json.
    profile = load_profile("org")
    profile["post_ledger"] = [p for p in profile["post_ledger"] if p["id"] != "post_002"]
    save_profile("org", profile)

    new_id = add_post("org", {"platform": "linkedin", "content": "new"})

    assert new_id == "post_004"
    ids = [p["id"] for p in load_profile("org")["post_ledger"]]
    assert len(ids) == len(set(ids))


def test_add_post_ignores_hand_written_non_standard_ids(tmp_data_dir):
    from agent.memory import add_post, load_profile, save_profile

    profile = load_profile("org")
    profile["post_ledger"] = [
        {"id": "spring-campaign", "platform": "facebook", "content": "x", "status": "posted"}
    ]
    save_profile("org", profile)
    assert add_post("org", {"platform": "linkedin", "content": "new"}) == "post_001"


def test_save_profile_keeps_backup_and_leaves_no_temp_file(tmp_data_dir):
    from agent.memory import load_profile, save_profile

    save_profile("org", {**load_profile("org"), "name": "First"})
    save_profile("org", {**load_profile("org"), "name": "Second"})
    org_dir = tmp_data_dir / "org"
    assert load_profile("org")["name"] == "Second"
    assert '"First"' in (org_dir / "profile.json.bak").read_text()
    # Exactly these two: the temporary file (named per process) was swapped in, not left behind.
    assert sorted(path.name for path in org_dir.iterdir()) == ["profile.json", "profile.json.bak"]


def test_load_profile_explains_hand_editing_mistake(tmp_data_dir):
    from agent.memory import ProfileError, load_profile

    (tmp_data_dir / "org").mkdir()
    (tmp_data_dir / "org" / "profile.json").write_text('{\n  "name": "x",\n}')
    with pytest.raises(ProfileError) as excinfo:
        load_profile("org")
    assert "line 3" in str(excinfo.value)
    assert "profile.json.bak" in str(excinfo.value)


def test_save_preference_records_quote_and_date(tmp_data_dir):
    from agent.memory import load_profile, save_preference

    save_preference("org", "voice", "warm and grassroots", quote="Too corporate, we're more warm")
    profile = load_profile("org")
    assert profile["preferences"]["voice"] == "warm and grassroots"
    note = profile["preference_notes"]["voice"]
    assert note["quote"] == "Too corporate, we're more warm"
    assert note["learned_at"]


def test_save_preference_append_builds_list_without_duplicates(tmp_data_dir):
    from agent.memory import load_profile, save_preference

    save_preference("org", "banned_topics", "politics")  # stored as a plain string first
    save_preference("org", "banned_topics", "religion", append=True)
    save_preference("org", "banned_topics", "religion", append=True)
    assert load_profile("org")["preferences"]["banned_topics"] == ["politics", "religion"]


def test_forget_preference_removes_key_and_note(tmp_data_dir):
    from agent.memory import forget_preference, load_profile, save_preference

    save_preference("org", "voice", "formal", quote="be formal")
    assert forget_preference("org", "voice") is True
    profile = load_profile("org")
    assert "voice" not in profile["preferences"]
    assert "voice" not in profile["preference_notes"]


def test_forget_preference_removes_single_list_item_ignoring_case(tmp_data_dir):
    from agent.memory import forget_preference, load_profile, save_preference

    save_preference("org", "banned_topics", "Politics", append=True)
    save_preference("org", "banned_topics", "religion", append=True)
    assert forget_preference("org", "banned_topics", "politics") is True
    assert load_profile("org")["preferences"]["banned_topics"] == ["religion"]
    assert forget_preference("org", "banned_topics", "religion") is True
    assert "banned_topics" not in load_profile("org")["preferences"]


def test_forget_preference_returns_false_when_nothing_matches(tmp_data_dir):
    from agent.memory import forget_preference, save_preference

    assert forget_preference("org", "voice") is False
    save_preference("org", "banned_topics", "politics", append=True)
    assert forget_preference("org", "banned_topics", "sports") is False


def test_forget_preference_keeps_a_string_preference_when_the_value_differs(tmp_data_dir):
    from agent.memory import forget_preference, load_profile, save_preference

    save_preference("org", "voice", "warm and grassroots")
    assert forget_preference("org", "voice", "corporate") is False
    assert load_profile("org")["preferences"]["voice"] == "warm and grassroots"
    assert forget_preference("org", "voice", "Warm and Grassroots") is True
    assert "voice" not in load_profile("org")["preferences"]


def test_add_voice_example_keeps_newest(tmp_data_dir):
    from agent.memory import MAX_VOICE_EXAMPLES, add_voice_example, load_profile

    for i in range(MAX_VOICE_EXAMPLES + 2):
        add_voice_example("org", f"post {i}", "pasted by user")
    examples = load_profile("org")["voice_examples"]
    assert len(examples) == MAX_VOICE_EXAMPLES
    assert examples[-1]["content"] == f"post {MAX_VOICE_EXAMPLES + 1}"
    assert examples[0]["source"] == "pasted by user"


def test_search_ledger_combines_filters(tmp_data_dir):
    from agent.memory import load_profile, save_profile, search_ledger

    profile = load_profile("org")
    profile["post_ledger"] = [
        {
            "id": "post_001",
            "platform": "instagram",
            "status": "approved",
            "content": "Gala recap",
            "created_at": "2026-09-28T10:00:00",
        },
        {
            "id": "post_002",
            "platform": "instagram",
            "status": "draft",
            "content": "Volunteer spotlight",
            "title": "Meet Ana",
            "created_at": "2026-10-02T10:00:00",
        },
        {
            "id": "post_003",
            "platform": "linkedin",
            "status": "draft",
            "content": "Annual report",
            "created_at": "2026-10-05T10:00:00",
        },
    ]
    save_profile("org", profile)

    def ids(**filters):
        return [p["id"] for p in search_ledger("org", **filters)]

    assert ids(status="draft") == ["post_002", "post_003"]
    assert ids(platform="Instagram") == ["post_001", "post_002"]
    assert ids(since="2026-10-01", until="2026-10-02") == ["post_002"]
    assert ids(text="meet ana") == ["post_002"]  # title is searched too
    assert ids(status="draft", platform="linkedin") == ["post_003"]
    assert ids() == ["post_001", "post_002", "post_003"]


def test_load_profile_rejects_a_profile_that_is_not_an_object(tmp_data_dir):
    from agent.memory import ProfileError, _profile_path, load_profile

    _profile_path("org").write_text("[]")
    with pytest.raises(ProfileError, match="JSON object"):
        load_profile("org")


def test_load_profile_rejects_a_ledger_that_is_not_a_list_of_posts(tmp_data_dir):
    from agent.memory import ProfileError, _profile_path, load_profile

    _profile_path("org").write_text('{"post_ledger": ["post_001"]}')
    with pytest.raises(ProfileError, match="post_ledger"):
        load_profile("org")


def test_load_profile_fills_in_keys_a_hand_edited_profile_lacks(tmp_data_dir):
    from agent.memory import _profile_path, load_profile

    _profile_path("org").write_text('{"name": "Org"}')
    profile = load_profile("org")
    assert profile["name"] == "Org"
    assert profile["post_ledger"] == [] and profile["preferences"] == {}


def test_hand_edited_post_without_fields_does_not_break_ledger_operations(tmp_data_dir):
    from agent.memory import _profile_path, add_post, search_ledger, update_post_status

    _profile_path("org").write_text('{"post_ledger": [{"title": "Typed by hand"}]}')
    assert search_ledger("org", platform="linkedin", since="2026-01-01") == []
    assert update_post_status("org", "post_001", "draft") is False
    assert add_post("org", {"platform": "linkedin", "content": "x"}) == "post_001"


@pytest.mark.parametrize(
    "section, value, expected",
    [
        ("preferences", None, "preferences must be a { ... } object"),
        ("preferences", ["warm"], "preferences must be a { ... } object"),
        ("preference_notes", "x", "preference_notes must be a { ... } object"),
        ("voice_examples", {"content": "x"}, "voice_examples must be a list"),
        ("voice_examples", ["just text"], "voice_examples must be a list"),
        ("research", "done", "research must be a { ... } object, or null"),
    ],
)
def test_load_profile_rejects_a_section_with_the_wrong_type(tmp_data_dir, section, value, expected):
    """Every turn reads these sections; a wrong type must be a ProfileError, not a crash later."""
    from agent.memory import ProfileError, load_profile

    (tmp_data_dir / "org").mkdir()
    (tmp_data_dir / "org" / "profile.json").write_text(json.dumps({section: value}))
    with pytest.raises(ProfileError) as excinfo:
        load_profile("org")
    assert expected in str(excinfo.value)
    assert "profile.json.bak" in str(excinfo.value)


def test_load_profile_accepts_null_research(tmp_data_dir):
    """The README tells users to set "research": null to research the org again."""
    from agent.memory import load_profile

    (tmp_data_dir / "org").mkdir()
    (tmp_data_dir / "org" / "profile.json").write_text('{"research": null}')
    assert load_profile("org")["research"] is None


def test_record_performance_stores_results_and_marks_posted(tmp_data_dir):
    from agent.memory import add_post, get_ledger, record_performance

    add_post("org", {"platform": "linkedin", "content": "x", "status": "approved"})
    assert record_performance("org", "post_001", {"reach": 900, "reactions": 40}, "2 donations")
    post = get_ledger("org")[0]
    assert post["status"] == "posted"
    assert post["performance"]["reach"] == 900
    assert post["performance"]["note"] == "2 donations"
    assert "recorded_at" in post["performance"]


def test_record_performance_replaces_earlier_results(tmp_data_dir):
    from agent.memory import add_post, get_ledger, record_performance

    add_post("org", {"platform": "linkedin", "content": "x"})
    record_performance("org", "post_001", {"reactions": 10}, "early")
    record_performance("org", "post_001", {"reactions": 50})
    assert get_ledger("org")[0]["performance"]["reactions"] == 50
    assert "note" not in get_ledger("org")[0]["performance"]


def test_record_performance_missing_post_returns_false(tmp_data_dir):
    from agent.memory import record_performance

    assert record_performance("org", "post_404", {"reactions": 1}) is False


def test_engagement_sums_engagement_metrics_but_not_reach():
    from agent.memory import engagement

    post = {"performance": {"reach": 1000, "reactions": 30, "comments": 5, "shares": 2}}
    assert engagement(post) == 37


@pytest.mark.parametrize(
    "performance",
    [None, "lots", {"reactions": "40"}, {"reactions": None}, {"reactions": True}],
)
def test_engagement_tolerates_hand_edited_results(performance):
    from agent.memory import engagement

    assert engagement({"performance": performance}) == 0


def test_top_performing_ranks_measured_posts_and_skips_the_rest():
    from agent.memory import top_performing

    ledger = [
        {"id": "post_001", "performance": {"reactions": 5}},
        {"id": "post_002"},
        {"id": "post_003", "performance": {"reactions": 50}},
        {"id": "post_004", "performance": {"shares": 20}},
    ]
    assert [p["id"] for p in top_performing(ledger, 2)] == ["post_003", "post_004"]
    assert top_performing([{"id": "post_002"}], 5) == []


def test_agenda_groups_posts_by_next_step():
    from datetime import date

    from agent.memory import agenda

    today = date(2026, 10, 19)
    ledger = [
        {"id": "post_001", "status": "planned", "scheduled_for": "2026-10-22"},
        {"id": "post_002", "status": "planned", "scheduled_for": "2026-11-20"},
        {"id": "post_003", "status": "draft", "platform": "linkedin"},
        {"id": "post_004", "status": "approved", "platform": "instagram"},
        {"id": "post_005", "status": "posted", "scheduled_for": "2026-10-10"},
        {"id": "post_006", "status": "posted", "scheduled_for": "2026-10-18"},
        {"id": "post_007", "status": "posted", "scheduled_for": "2026-10-01", "performance": {}},
    ]
    groups = agenda(ledger, today)
    assert [p["id"] for p in groups["to_draft"]] == ["post_001"]
    assert [p["id"] for p in groups["awaiting_approval"]] == ["post_003"]
    assert [p["id"] for p in groups["to_publish"]] == ["post_004"]
    assert [p["id"] for p in groups["needs_results"]] == ["post_005"]


def test_agenda_skips_drafts_replaced_on_the_same_platform_only():
    """A rejected rewrite doesn't replace its original: that draft still awaits approval."""
    from datetime import date

    from agent.memory import agenda

    ledger = [
        {"id": "post_001", "status": "draft", "platform": "linkedin"},
        {"id": "post_002", "status": "draft", "platform": "linkedin", "parent_id": "post_001"},
        {"id": "post_003", "status": "draft", "platform": "instagram", "parent_id": "post_002"},
        {"id": "post_004", "status": "rejected", "platform": "instagram", "parent_id": "post_003"},
    ]
    waiting = agenda(ledger, date(2026, 10, 19))["awaiting_approval"]
    assert [p["id"] for p in waiting] == ["post_002", "post_003"]


def test_agenda_is_empty_for_an_empty_or_hand_edited_ledger():
    from datetime import date

    from agent.memory import agenda

    assert not any(agenda([], date(2026, 10, 19)).values())
    assert not any(agenda([{"id": "post_001"}], date(2026, 10, 19)).values())


@pytest.mark.parametrize("name", ["!!!", "  ", "—", "🎉🎉", "-_-"])
def test_slugify_gives_no_id_to_a_name_without_letters_or_digits(name):
    from agent.memory import slugify

    assert slugify(name) == ""


def test_fill_org_details_sets_a_new_orgs_name_and_website(tmp_data_dir):
    from agent.memory import fill_org_details, load_profile

    fill_org_details("food-bank", "Food Bank", "https://food.example")
    profile = load_profile("food-bank")
    assert (profile["name"], profile["website"]) == ("Food Bank", "https://food.example")


def test_fill_org_details_keeps_what_the_profile_already_has(tmp_data_dir):
    from agent.memory import fill_org_details, load_profile, save_profile

    save_profile("food-bank", load_profile("food-bank") | {"name": "Food Bank of Ohio"})
    fill_org_details("food-bank", "food bank", "https://food.example")
    fill_org_details("food-bank", "Other", "https://other.example")
    profile = load_profile("food-bank")
    assert (profile["name"], profile["website"]) == ("Food Bank of Ohio", "https://food.example")


def test_locked_is_reentrant_so_a_locked_caller_can_add_a_post(tmp_data_dir):
    """flock blocks a second open even in the same process; reentrancy prevents a deadlock."""
    import threading

    from agent.memory import add_post, get_ledger, locked

    def nested() -> None:
        with locked("org"), locked("org"):
            add_post("org", {"platform": "general", "content": "x"})

    worker = threading.Thread(target=nested, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "locked() deadlocked on itself"
    assert len(get_ledger("org")) == 1


def test_locked_reentrancy_is_per_thread_so_other_threads_still_wait(tmp_data_dir):
    """Only the holding thread may re-enter; another thread must still wait for the lock."""
    import threading

    from agent.memory import add_post, get_ledger, locked

    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with locked("org"):
            holding.set()
            release.wait(5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holding.wait(5)
    writer = threading.Thread(
        target=add_post, args=("org", {"platform": "general", "content": "x"}), daemon=True
    )
    writer.start()
    writer.join(0.2)  # a writer that skipped the lock is done in milliseconds
    assert writer.is_alive() and get_ledger("org") == []
    release.set()
    writer.join(5)
    assert len(get_ledger("org")) == 1
