"""
Deployment check: the Docker Compose stack, as `docker compose up` starts it, in a real browser.

Owns: that a freshly deployed stack serves the login page, refuses a wrong
password, lets the default account (created by the migrate service from
MACKIE_DEFAULT_PASSWORD) log in, and onboards an organization: the database
roles, the schema, the app's database connection, the login secret and
Chainlit all work together. No model is called. The rest of the deployment
(the migrate service exiting cleanly, the app not running as root, the
database not published, the app's role not privileged) is checked by the CI
job's shell steps
(.github/workflows/ci.yml). Does not own the app's behaviour (tests/browser/,
tests/test_ui.py).

Opt-in: marked `deployed` and run against a stack that is already up:
    MACKIE_DEPLOYED_URL=http://127.0.0.1:8000 MACKIE_DEPLOYED_PASSWORD=... pytest -m deployed
"""

import os
import uuid

import pytest

# In CI the check must run: a missing playwright fails instead of skipping the job green.
if os.getenv("MACKIE_DEPLOYED_REQUIRED"):
    import playwright.sync_api as _sync_api  # noqa: F401

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.deployed

URL: str = os.getenv("MACKIE_DEPLOYED_URL", "http://127.0.0.1:8000")
USERNAME: str = os.getenv("MACKIE_DEPLOYED_USERNAME", "admin")
WAIT_MS: int = 30_000


@pytest.fixture(scope="module")
def password() -> str:
    value = os.getenv("MACKIE_DEPLOYED_PASSWORD", "")
    if not value:
        # In CI a missing password means the job is broken: skipping would pass it green.
        if os.getenv("MACKIE_DEPLOYED_REQUIRED"):
            pytest.fail(
                "MACKIE_DEPLOYED_PASSWORD is empty: the deployment check has nothing to log in with"
            )
        pytest.skip("MACKIE_DEPLOYED_PASSWORD is not set: there is no deployed stack to check")
    return value


@pytest.fixture
def page():
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        yield context.new_page()
        context.close()
        browser.close()


def _log_in(page, password: str) -> None:
    page.goto(URL)
    page.locator("#email").fill(USERNAME, timeout=WAIT_MS)
    page.locator("#password").fill(password)
    page.get_by_role("button", name="Sign In").click()


def test_a_wrong_password_is_refused(page, password):
    _log_in(page, password + "-wrong")
    page.get_by_text("Unable to sign in").wait_for(timeout=WAIT_MS)


def test_the_default_account_logs_in_and_onboards_an_organization(page, password):
    _log_in(page, password)
    page.get_by_text("What organization are you working with?").wait_for(timeout=WAIT_MS)
    org = f"Deploy Check {uuid.uuid4().hex[:8]}"  # new every run, so the greeting is a new org's
    box = page.locator("#chat-input")
    box.click()
    box.fill(org)
    page.keyboard.press("Enter")
    page.get_by_role("button", name="Yes, that's us").click(timeout=WAIT_MS)
    page.get_by_text("set up for").wait_for(timeout=WAIT_MS)
    assert org in page.inner_text("body")
