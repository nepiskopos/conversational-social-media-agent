"""
Tests for agent/publishing.py: connected accounts (saved, replaced, removed,
refused when they can't be used), the permissions each publisher needs, the
dummy publishers, and publishing a post to a platform at most once: recorded,
not repeated, retried after a failure or a timeout or a claim a stopped process
left, and never retried while another session holds the claim. When a post may
be published is owned by test_tools.py; publishing from parallel sessions by
test_sessions.py, in parallel by test_performance.py, across orgs by
test_security.py.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

LINKEDIN = ("linkedin", "urn:li:organization:1", "Food Bank", ("w_organization_social",))
REQUEST_ARGS = ("post_001", "Our post #a #b #c", "A photo")


async def _org_with_account(platform: tuple = LINKEDIN):
    from agent import memory as mem
    from agent import publishing

    await mem.fill_org_details("org", "Org")
    account = publishing.PlatformAccount(*platform)
    await publishing.connect_account("org", account)
    return account


async def test_connect_get_and_disconnect_an_account(database):
    from agent import publishing

    account = await _org_with_account()
    assert await publishing.get_accounts("org") == {"linkedin": account}
    replaced = publishing.PlatformAccount("linkedin", "urn:li:organization:2", "New", ())
    await publishing.connect_account("org", replaced)
    assert await publishing.get_accounts("org") == {"linkedin": replaced}
    assert await publishing.disconnect_account("org", "linkedin") is True
    assert await publishing.disconnect_account("org", "linkedin") is False
    assert await publishing.get_accounts("org") == {}


@pytest.mark.parametrize(
    "platform, account_id, org, message",
    [
        ("email", "list-1", "org", "can't publish to 'email'"),
        ("linkedin", "", "org", "without spaces"),
        ("linkedin", "my page", "org", "without spaces"),
        ("linkedin", "urn:1", "nobody", "There is no organization nobody"),
    ],
)
async def test_an_account_that_cant_be_used_is_refused(
    database, platform, account_id, org, message
):
    from agent import memory as mem
    from agent import publishing

    await mem.fill_org_details("org", "Org")
    with pytest.raises(publishing.PublishingError, match=message):
        await publishing.connect_account(org, publishing.PlatformAccount(platform, account_id))
    assert await publishing.get_accounts("org") == {}


def test_missing_scopes_lists_the_permissions_the_publisher_needs():
    from agent.publishing import PUBLISHERS, PlatformAccount, missing_scopes

    assert set(PUBLISHERS) == {"linkedin", "facebook", "instagram"}
    instagram = PlatformAccount("instagram", "1", scopes=("instagram_basic",))
    assert missing_scopes(instagram) == ["instagram_content_publish"]
    assert missing_scopes(PlatformAccount(*LINKEDIN)) == []


async def test_the_dummy_publisher_sends_nothing_and_says_so():
    from agent.publishing import PUBLISHERS, PlatformAccount, PublishRequest

    request = PublishRequest(*REQUEST_ARGS)
    receipt = await PUBLISHERS["linkedin"].publish(request, PlatformAccount(*LINKEDIN))
    assert receipt.simulated is True and receipt.url == ""
    assert receipt.external_id == "simulated-linkedin-urn:li:organization:1-post_001"


async def test_a_post_is_published_to_a_platform_once(database):
    from agent import publishing

    account = await _org_with_account()
    request = publishing.PublishRequest(*REQUEST_ARGS)
    [first] = await publishing.publish("org", request, [account])
    [again] = await publishing.publish("org", request, [account])
    assert first.published and first.message.startswith("published (simulated: ")
    assert not again.published and again.message.startswith("already published on ")
    assert "(simulated-linkedin-urn:li:organization:1-post_001)" in again.message
    assert await publishing.published_platforms("org", "post_001") == ["linkedin"]
    row = database["cleaner"].execute("SELECT state, simulated, error FROM publications").fetchone()
    assert row == ("published", True, "")


async def test_a_failed_publication_is_recorded_and_can_be_tried_again(database):
    from agent import publishing

    account = await _org_with_account()
    request = publishing.PublishRequest(*REQUEST_ARGS)
    failing = AsyncMock(side_effect=RuntimeError("platform down"))
    with patch.object(publishing.DummyPublisher, "publish", failing):
        [failed] = await publishing.publish("org", request, [account])
    assert not failed.published
    assert failed.message.startswith("failed (RuntimeError); nothing is recorded")
    assert await publishing.published_platforms("org", "post_001") == []
    [retried] = await publishing.publish("org", request, [account])
    assert retried.published


async def test_a_publisher_that_never_answers_times_out(database):
    """Nothing waits forever: a hung platform call becomes a failure the user can retry."""
    from agent import publishing

    account = await _org_with_account()
    never = asyncio.Event()

    async def hang(self, request, account):
        await never.wait()

    with (
        patch.object(publishing.DummyPublisher, "publish", hang),
        patch("agent.publishing.PUBLISH_TIMEOUT_S", 0.05),
    ):
        [outcome] = await publishing.publish(
            "org", publishing.PublishRequest(*REQUEST_ARGS), [account]
        )
    assert outcome.message.startswith("failed (TimeoutError)")


@pytest.mark.parametrize("age_s, claimed", [(0, False), (10_000, True)])
async def test_a_claim_is_taken_over_only_once_it_is_stale(database, age_s, claimed):
    """A claim held right now is another session's; an old one was left by a stopped process."""
    from agent import publishing

    account = await _org_with_account()
    database["cleaner"].execute(
        "INSERT INTO publications (org_id, post_id, platform, state, account_id, updated_at) "
        "VALUES ('org', 'post_001', 'linkedin', 'publishing', 'x', now() - make_interval(secs => %s))",
        (age_s,),
    )
    [outcome] = await publishing.publish("org", publishing.PublishRequest(*REQUEST_ARGS), [account])
    assert outcome.published is claimed
    if not claimed:
        assert outcome.message == "another session is publishing it right now"


async def test_one_platform_failing_doesnt_stop_the_others(database):
    from agent import publishing

    linkedin = await _org_with_account()
    facebook = publishing.PlatformAccount("facebook", "123", "Page", ("pages_manage_posts",))
    await publishing.connect_account("org", facebook)
    real = publishing.DummyPublisher.publish

    async def fail_on_facebook(self, request, account):
        if self.platform == "facebook":
            raise ConnectionError("down")
        return await real(self, request, account)

    with patch.object(publishing.DummyPublisher, "publish", fail_on_facebook):
        outcomes = await publishing.publish(
            "org", publishing.PublishRequest(*REQUEST_ARGS), [linkedin, facebook]
        )
    assert [(o.platform, o.published) for o in outcomes] == [
        ("linkedin", True),
        ("facebook", False),
    ]
    assert await publishing.published_platforms("org", "post_001") == ["linkedin"]
