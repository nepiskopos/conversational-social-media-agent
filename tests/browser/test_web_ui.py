"""
Browser tests: the real web UI, driven in headless Chromium.

Owns: what a user of the web UI sees, end to end through the real Chainlit
server and frontend: onboarding, the greeting, streamed replies, tool steps,
a returning org after a reload, the layout at phone width, and two browsers
on one org keeping their conversations apart. The model is scripted
(serve.py), so no API calls are made. Does not own app.py's logic in detail
(test_ui.py, with a fake Chainlit, covers every branch in milliseconds).

Opt-in: these are marked `browser` and skipped by a plain `pytest` run. Run
them with `pytest -m browser` after `playwright install chromium` (playwright
itself is in requirements-dev.txt).
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.browser.serve import FIRST_SENTENCE, RELEASE_FILE_NAME, SECOND_SENTENCE

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
def server(tmp_path_factory) -> Iterator[tuple[str, Path]]:
    """The real web UI on a free port with a scratch data folder; yields (url, data_dir)."""
    data_dir, port = tmp_path_factory.mktemp("data"), _free_port()
    log = (data_dir / "server.log").open("w")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "serve.py"), str(data_dir), str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
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


def _open(browser, url: str, viewport: dict[str, int] = DESKTOP):
    """A new browser profile (its own cookies, so a separate user) on the web UI."""
    page = browser.new_context(viewport=viewport).new_page()
    page.console_errors = []
    page.on("console", lambda m: page.console_errors.append(m.text) if m.type == "error" else None)
    page.goto(url)
    return page


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


def test_a_new_user_is_onboarded_greeted_and_answered(server, browser):
    url, _ = server
    page = _open(browser, url)
    _onboard(page, "Harbor Food Bank, https://harbor.example")

    _wait_for_text(page, "set up for")
    assert "Harbor Food Bank" in page.inner_text("body")
    _send(page, "hello")
    _wait_for_text(page, "You said: hello. Messages I can see: 1.")
    assert page.console_errors == []


def test_a_reply_appears_while_it_is_still_streaming(server, browser):
    """The second sentence is held back until the test releases it; the first must show already."""
    url, data_dir = server
    page = _open(browser, url)
    _onboard(page, "Stream Org")
    _wait_for_text(page, "set up for")

    _send(page, "stream slowly")
    _wait_for_text(page, FIRST_SENTENCE)
    assert SECOND_SENTENCE.strip() not in page.inner_text("body")
    (data_dir / RELEASE_FILE_NAME).touch()
    _wait_for_text(page, SECOND_SENTENCE.strip())


def test_a_saved_post_shows_its_tool_step_and_is_remembered_after_a_reload(server, browser):
    url, data_dir = server
    page = _open(browser, url)
    _onboard(page, "Reload Org")
    _wait_for_text(page, "set up for")

    _send(page, "save Gala recap")
    _wait_for_text(page, "Saved it to the ledger.")
    assert "save_post" in page.inner_text("body")
    ledger = json.loads((data_dir / "reload-org" / "profile.json").read_text())["post_ledger"]
    assert [post["title"] for post in ledger] == ["Gala recap"]

    page.reload()
    _onboard(page, "Reload Org")
    _wait_for_text(page, "Welcome back!")
    assert "1 post for" in page.inner_text("body")


def test_the_chat_fits_a_phone_screen(server, browser):
    url, _ = server
    page = _open(browser, url, PHONE)
    page.get_by_text("What organization are you working with?").wait_for(timeout=WAIT_MS)

    def fits_the_screen(locator) -> bool:
        box = locator.bounding_box()
        return box is not None and box["x"] >= 0 and box["x"] + box["width"] <= PHONE["width"]

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


def test_two_browsers_on_one_org_keep_their_conversations_apart(server, browser):
    url, _ = server
    alice, bob = _open(browser, url), _open(browser, url)
    for page in (alice, bob):
        _onboard(page, "Shared Org")
        _wait_for_text(page, "set up for")

    _send(alice, "secret plan from Alice")
    _wait_for_text(alice, "Messages I can see: 1.")
    _send(bob, "hello from Bob")
    _wait_for_text(bob, "You said: hello from Bob. Messages I can see: 1.")
    assert "secret plan" not in bob.inner_text("body")

    _send(alice, "and a second message")
    _wait_for_text(alice, "Messages I can see: 2.")  # Alice keeps her own history
