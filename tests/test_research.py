"""
Unit tests for agent/research.py: website scraping and web search.

Owns: scrape_site (page text, the per-page cap, subpages, redirects, failures),
the basic SSRF blocklist of _is_safe_url and the DNS check, _resolve_host,
_find_subpages, search_web and research_org (queries, sources). Does not own the
disguised-host SSRF bypasses (test_security.py) or the parallelism of research
(test_performance.py).
"""

import asyncio
import gzip
import zlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Captured at import, before the autouse no_real_dns fixture replaces it.
from agent.research import _resolve_host as _real_resolve_host


def _response(
    text: str = "",
    location: str | None = None,
    error: Exception | None = None,
    encoded: tuple[str, bytes] | None = None,
) -> MagicMock:
    """
    Build a mock streamed httpx response; a location makes it a redirect.

    encoded is (Content-Encoding, the body as sent) for a compressed page.
    """

    async def body() -> AsyncIterator[bytes]:
        yield encoded[1] if encoded else text.encode()

    response = MagicMock()
    response.aiter_raw = body
    response.encoding = "utf-8"
    response.is_redirect = location is not None
    response.headers = {"location": location} if location else {}
    if encoded:
        response.headers["content-encoding"] = encoded[0]
    response.raise_for_status = MagicMock(side_effect=error)
    return response


def _fake_client(get: Callable[..., Any] | list) -> MagicMock:
    """
    An httpx client whose stream("GET", url, **kwargs) serves get(url, **kwargs).

    get is a function or a list of responses (or exceptions to raise) in order.
    client.get records every request, so tests check the URLs and arguments.
    """
    client = MagicMock()
    client.get = MagicMock(side_effect=get)

    @asynccontextmanager
    async def stream(method: str, url: str, **kwargs: Any) -> AsyncIterator[MagicMock]:
        yield client.get(url, **kwargs)

    client.stream = stream
    return client


async def _scrape(url: str, client: MagicMock) -> tuple[str, list[str]]:
    """Run scrape_site with httpx replaced by client."""
    from agent.research import scrape_site

    with patch("agent.research.httpx.AsyncClient") as mock_class:
        mock_class.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
        return await scrape_site(url)


async def _scrape_with_responses(url: str, responses: list) -> tuple[str, list[str], MagicMock]:
    """Run scrape_site with httpx returning the given responses (or raising them) in order."""
    client = _fake_client(responses)
    text, urls = await _scrape(url, client)
    return text, urls, client


async def test_scrape_site_keeps_body_text_and_drops_scripts_styles_and_site_chrome():
    result, urls, _ = await _scrape_with_responses(
        "https://example.com",
        [
            _response(
                "<html><head><style>.style-rule { color: red; }</style></head><body>"
                "<header>Header banner</header><nav>Nav menu</nav>"
                "<p>Mission: help kids.</p>"
                "<script>script code</script>"
                "<footer>Footer links</footer></body></html>"
            )
        ],
    )
    assert "Mission: help kids." in result
    for dropped in ("style-rule", "Header banner", "Nav menu", "script code", "Footer links"):
        assert dropped not in result
    assert urls == ["https://example.com"]


async def test_scrape_site_caps_the_text_of_each_page():
    from agent.research import SITE_PAGE_CHARS

    result, _, _ = await _scrape_with_responses(
        "https://example.com", [_response("<p>" + "a" * 10_000 + "</p>")]
    )
    assert result == "[Page: https://example.com]\n" + "a" * SITE_PAGE_CHARS


@pytest.mark.parametrize(
    "given, fetched",
    [("example.com", "https://example.com"), ("HTTPS://example.com", "HTTPS://example.com")],
    ids=["no-scheme", "uppercase-scheme"],
)
async def test_scrape_site_adds_https_only_when_the_url_has_no_scheme(given, fetched):
    _, _, client = await _scrape_with_responses(given, [_response("<p>Content</p>")])
    assert client.get.call_args[0][0] == fetched


@pytest.mark.parametrize(
    "response",
    [Exception("timeout"), _response("Not Found", error=Exception("404 Not Found"))],
    ids=["get-raises", "error-status"],
)
async def test_scrape_site_reports_a_homepage_it_could_not_fetch(response):
    result, urls, _ = await _scrape_with_responses("https://broken.example.com", [response])
    assert (result, urls) == ("", [])  # an error's text is no research; the log has it


async def test_search_web_returns_formatted_results():
    from agent.research import FETCH_TIMEOUT_S, search_web

    mock_response = {
        "results": [
            {"title": "Title 1", "url": "https://a.com", "content": "Snippet 1"},
            {"title": "Title 2", "url": "https://b.com"},  # no snippet: still a result
            {"title": "No URL", "content": "nothing to cite"},  # dropped
        ]
    }
    with patch("agent.research.AsyncTavilyClient") as mock_class:
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(return_value=mock_response)
        mock_class.return_value = mock_instance
        results = await search_web("test query", max_results=2)
    assert len(results) == 2
    assert results[0]["title"] == "Title 1"
    assert results[0]["url"] == "https://a.com"
    assert results[0]["snippet"] == "Snippet 1"
    assert results[1] == {"title": "Title 2", "url": "https://b.com", "snippet": ""}
    # A stalled search must not hold up the reply.
    mock_instance.search.assert_awaited_once_with(
        "test query", max_results=2, timeout=FETCH_TIMEOUT_S
    )


async def test_search_web_gives_up_on_a_search_that_never_finishes(monkeypatch):
    """Tavily's timeout bounds each wait for data, not a search that trickles forever."""
    from agent import research

    monkeypatch.setattr(research, "FETCH_TIMEOUT_S", 0.01)
    with patch("agent.research.AsyncTavilyClient") as mock_class:
        mock_class.return_value.search = AsyncMock(side_effect=asyncio.Event().wait)
        assert await asyncio.wait_for(research.search_web("q"), 5) == []


@pytest.mark.parametrize(
    "typed, url",
    [
        ("myorg.org", "https://myorg.org"),
        ("  http://www.myorg.org/about ", "http://www.myorg.org/about"),
        ("xn--bcher-kva.example", "https://xn--bcher-kva.example"),
        ("Inc.", ""),
        ("our site", ""),
        ("localhost", ""),
        ("http://127.0.0.1", ""),
        ("ftp://myorg.org", ""),
        ("", ""),
        ("(myorg.org).", "https://myorg.org"),  # punctuation around it
        ("café-lumière.fr", "https://café-lumière.fr"),  # checked in its IDNA form
        ("myorg.org:99999", ""),  # not a port
        ("https://user:pw@evil.example", ""),  # a login hides the host from a reader
    ],
)
def test_website_url_takes_only_a_website(typed, url):
    from agent.research import website_url

    assert website_url(typed) == url


@pytest.mark.parametrize(
    "url, expected",
    [
        ("org.example", "https://org.example"),
        ("HTTP://org.example", "HTTP://org.example"),
        ("evil.example://org.example", "https://evil.example://org.example"),
    ],
)
def test_with_scheme_adds_https_to_anything_without_an_http_scheme(url, expected):
    from agent.research import with_scheme

    assert with_scheme(url) == expected


async def test_search_web_handles_exception():
    from agent.research import search_web

    with patch("agent.research.AsyncTavilyClient") as mock_class:
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(side_effect=Exception("api error"))
        mock_class.return_value = mock_instance
        results = await search_web("test query")
    assert results == []


async def test_research_org_with_url_scrapes_once_searches_three_times_and_cites_sources():
    from datetime import date

    from agent.research import research_org

    social = [
        {"title": "FB Page", "url": "https://fb.com/org", "snippet": "Social"},
    ]
    news = [{"title": "News Item", "url": "https://news.com/article", "snippet": "News"}]
    similar = [{"title": "Similar Org", "url": "https://sim.org", "snippet": "Similar"}]
    with (
        patch(
            "agent.research.scrape_site",
            new=AsyncMock(
                return_value=("scraped content", ["https://test.org", "https://test.org/programs"])
            ),
        ) as mock_scrape,
        patch(
            "agent.research.search_web", new=AsyncMock(side_effect=[social, news, similar])
        ) as mock_search,
    ):
        year_before = date.today().year
        result = await research_org("Test Org", "https://test.org")

    mock_scrape.assert_called_once_with("https://test.org")
    assert mock_search.call_count == 3
    queries = [c.args[0] for c in mock_search.call_args_list]
    # Either year: the test may run across New Year.
    assert any(str(year) in queries[1] for year in (year_before, date.today().year))
    assert result == {
        "org_name": "Test Org",
        "website_content": "scraped content",
        "social_presence": social,
        "news": news,
        "similar_orgs": similar,
        "sources": [  # similar orgs are context, not citations
            {"type": "website", "url": "https://test.org"},
            {"type": "website", "url": "https://test.org/programs"},
            {"type": "social", "url": "https://fb.com/org", "title": "FB Page"},
            {"type": "news", "url": "https://news.com/article", "title": "News Item"},
        ],
    }
    assert all(source["url"] for source in result["sources"])


async def test_research_org_without_url_skips_scrape():
    from agent.research import research_org

    search_result = [{"title": "T", "url": "https://x.com", "snippet": "S"}]
    with (
        patch("agent.research.scrape_site", new=AsyncMock()) as mock_scrape,
        patch(
            "agent.research.search_web", new=AsyncMock(return_value=search_result)
        ) as mock_search,
    ):
        result = await research_org("Test Org")

    mock_scrape.assert_not_called()
    assert mock_search.call_count == 3
    assert result["website_content"] == ""


@pytest.mark.parametrize(
    "url",
    [
        "https://brcastrong.org",
        "http://example.com",
        # "127.0.0.1@" is userinfo; the host is example.com, which DNS checks later.
        "http://127.0.0.1@example.com/",
        "https://org.example/about",
        "https://bücher.example/",  # internationalised domain
        "http://[64:ff9b::808:808]/",  # NAT64 to a public address (8.8.8.8)
        "http://[2606:4700:4700::1111]/",  # public IPv6
    ],
)
def test_is_safe_url_allows_public_hosts(url):
    from agent.research import _is_safe_url

    assert _is_safe_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1:8080",
        "http://127.255.255.255",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "http://metadata.google.internal/",  # GCP metadata
        "http://100.100.100.200/",  # Alibaba Cloud metadata
        "http://10.0.0.1",
        "http://192.168.1.1",
        "http://172.16.0.1",
        "http://100.64.0.1/",  # carrier-grade NAT
        "http://0.0.0.0/",  # unspecified, reaches localhost on Linux
        "http://[::]/",
        "http://[::1]/",  # IPv6 loopback
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "http://[fc00::1]/",  # IPv6 unique local
        "http://[fe80::1]/",  # IPv6 link-local
        "http://0x7f000001/",  # hex form of 127.0.0.1
        "http://2130706433/",  # decimal form of 127.0.0.1
        "http://127.1/",  # short form of 127.0.0.1
        "http://0177.0.0.1/",  # octal form of 127.0.0.1
        "HTTP://127.0.0.1/",  # uppercase scheme
        "http://user@127.0.0.1/",  # userinfo before the real host
        "http://localhost./",  # trailing dot
        "http://admin.localhost/",
        "file:///etc/passwd",
        "ftp://files.example.com",
        "javascript:alert(1)",
    ],
)
def test_is_safe_url_blocks_internal_hosts_and_non_http_schemes(url):
    """The basics; disguised and embedded forms (NAT64, encoded hosts) are in test_security.py."""
    from agent.research import _is_safe_url

    assert _is_safe_url(url) is False


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",  # cloud metadata
        "100.100.100.200",  # carrier-grade NAT (Alibaba Cloud metadata)
        # Regression: IPv6 forms of internal addresses, as a DNS64 resolver may answer.
        "64:ff9b::a9fe:a9fe",
        "::ffff:10.0.0.1",
        "fd00::1",
    ],
)
async def test_scrape_site_blocks_a_domain_that_resolves_to_an_internal_address(address):
    with patch("agent.research._resolve_host", new=AsyncMock(return_value=[address])):
        result, _, client = await _scrape_with_responses("https://innocent-looking.example", [])
    assert result == ""  # refused; the log says why (test_logging.py)
    client.get.assert_not_called()


@pytest.mark.parametrize(
    "location", ["file:///etc/passwd", "http://169.254.169.254/latest/meta-data/"]
)
async def test_scrape_site_does_not_follow_a_redirect_to_a_blocked_url(location):
    result, _, client = await _scrape_with_responses(
        "https://example.org", [_response(location=location)]
    )
    assert result == ""  # refused; the log says why (test_logging.py)
    assert client.get.call_count == 1


@pytest.mark.parametrize("url", ["http://169.254.169.254/", "http://localhost:6379/"])
async def test_scrape_site_refuses_a_blocked_url_without_fetching(url):
    result, urls, client = await _scrape_with_responses(url, [])
    assert result == ""  # refused; the log says why (test_logging.py)
    assert urls == []
    client.get.assert_not_called()


async def test_scrape_site_follows_safe_redirect():
    result, _, client = await _scrape_with_responses(
        "http://example.org",
        [
            _response(location="https://www.example.org/"),
            _response("<html><body><p>Our mission</p></body></html>"),
        ],
    )
    assert "Our mission" in result
    assert client.get.call_args_list[1][0][0] == "https://www.example.org/"
    # httpx must never follow a redirect itself: each hop is checked here first.
    assert all(c.kwargs["follow_redirects"] is False for c in client.get.call_args_list)


async def test_scrape_site_resolves_relative_redirect():
    _, _, client = await _scrape_with_responses(
        "https://example.org/old",
        [
            _response(location="/new"),
            _response("<p>ok</p>"),
        ],
    )
    assert client.get.call_args_list[1][0][0] == "https://example.org/new"


async def test_scrape_site_stops_after_too_many_redirects():
    from agent.research import MAX_REDIRECTS

    loop = [_response(location="https://example.org/loop") for _ in range(MAX_REDIRECTS + 1)]
    result, _, _ = await _scrape_with_responses("https://example.org", loop)
    assert result == ""  # given up; the log says why


async def test_scrape_site_blocks_redirect_to_domain_that_resolves_to_loopback():
    resolver = AsyncMock(side_effect=[["93.184.216.34"], ["::ffff:127.0.0.1"]])
    with patch("agent.research._resolve_host", new=resolver):
        result, _, client = await _scrape_with_responses(
            "https://example.org",
            [
                _response(location="https://rebound.example/"),
            ],
        )
    assert result == ""  # refused; the log says why (test_logging.py)
    assert client.get.call_count == 1


async def test_resolve_host_returns_addresses_and_empty_on_failure():
    import socket

    loop = MagicMock()
    loop.getaddrinfo = AsyncMock(return_value=[(None, None, None, "", ("203.0.113.5", 0))])
    with patch("agent.research.asyncio.get_running_loop", return_value=loop):
        assert await _real_resolve_host("org.example") == ["203.0.113.5"]
        loop.getaddrinfo = AsyncMock(side_effect=socket.gaierror("no such host"))
        assert await _real_resolve_host("missing.example") == []


HOME_HTML = """<html><body>
<nav><a href="/about-us">About</a> <a href="/programs/">Programs</a> <a href="/donate">Donate</a></nav>
<a href="https://twitter.com/x/about">Our Twitter</a>
<a href="/files/annual-impact-report.pdf">Report</a>
<a href="/programs/#top">Programs again</a>
<p>Welcome to Test Org.</p>
</body></html>"""


def test_find_subpages_picks_same_site_descriptive_pages():
    from agent.research import _find_subpages

    pages = _find_subpages(HOME_HTML, "https://www.test.org/")
    assert pages == ["https://www.test.org/about-us", "https://www.test.org/programs/"]


async def test_a_fetch_reads_at_most_max_page_bytes_of_an_endless_page(monkeypatch):
    """A page that never ends must not fill the server's memory."""
    from agent import research

    monkeypatch.setattr(research, "MAX_PAGE_BYTES", 1000)

    async def endless() -> AsyncIterator[bytes]:
        while True:
            yield b"<p>" + b"x" * 300 + b"</p>"

    response = _response()
    response.aiter_raw = endless
    _, html = await _scrape_html(_fake_client([response]))
    assert len(html) == 1000


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
async def test_a_compressed_page_is_unpacked_only_up_to_max_page_bytes(monkeypatch, encoding):
    """Regression: httpx unpacked whole chunks, so 2 KB of gzip became 1 GB in memory."""
    import zlib

    from agent import research

    monkeypatch.setattr(research, "MAX_PAGE_BYTES", 1000)
    packer = zlib.compressobj(wbits=31 if encoding == "gzip" else 15)
    bomb = packer.compress(b"<p>" + b"x" * 10_000_000) + packer.flush()
    assert len(bomb) < 20_000
    client = _fake_client([_response(encoded=(encoding, bomb))])
    _, html = await _scrape_html(client)
    assert len(html) == 1000 and html.startswith("<p>xxx")
    sent = client.get.call_args.kwargs["headers"]["Accept-Encoding"]
    assert sent == "identity, gzip, deflate"


@pytest.mark.parametrize(
    "encoding, body",
    [
        ("none", b"<p>plain</p>"),  # misconfigured servers send these; httpx reads them as plain
        ("utf-8", b"<p>plain</p>"),
        ("deflate", zlib.compress(b"<p>plain</p>")[2:-4]),  # raw deflate, without zlib's wrapper
    ],
)
async def test_a_page_in_an_odd_encoding_is_still_read(encoding, body):
    _, html = await _scrape_html(_fake_client([_response(encoded=(encoding, body))]))
    assert html == "<p>plain</p>"


@pytest.mark.parametrize("wbits", [zlib.MAX_WBITS, -zlib.MAX_WBITS], ids=["zlib", "raw"])
async def test_a_deflate_page_is_unpacked_across_its_chunks(wbits):
    """The first chunk decides zlib-wrapped or raw; the rest go through the same unpacker."""
    packer = zlib.compressobj(wbits=wbits)
    body = packer.compress(b"<p>" + b"plain " * 200 + b"</p>") + packer.flush()

    async def in_three_chunks() -> AsyncIterator[bytes]:
        third = len(body) // 3 + 1
        for start in range(0, len(body), third):
            yield body[start : start + third]

    response = _response(encoded=("deflate", b""))
    response.aiter_raw = in_three_chunks
    _, html = await _scrape_html(_fake_client([response]))
    assert html == "<p>" + "plain " * 200 + "</p>"


async def test_bytes_after_the_end_of_a_compressed_page_are_not_read(monkeypatch):
    """Regression: endless bytes after a tiny gzip stream piled up, slower with every chunk."""
    from agent import research

    monkeypatch.setattr(research, "MAX_PAGE_BYTES", 1000)
    page = gzip.compress(b"<p>hi</p>")
    chunks_read = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal chunks_read
        yield page
        while True:
            chunks_read += 1
            yield b"\0" * 300

    response = _response(encoded=("gzip", b""))
    response.aiter_raw = endless
    _, html = await _scrape_html(_fake_client([response]))
    assert html == "<p>hi</p>" and chunks_read == 0


@pytest.mark.parametrize("encoding", ["br", "gzip, gzip", "zstd"])
async def test_a_page_in_an_encoding_mackie_didnt_ask_for_is_refused(encoding):
    with pytest.raises(ValueError, match="Content-Encoding"):
        await _scrape_html(_fake_client([_response(encoded=(encoding, b"\x1f\x8b..."))]))


async def _scrape_html(client: MagicMock) -> tuple[str, str]:
    """research._fetch_html with httpx replaced by client (see _fake_client)."""
    from agent import research

    with patch("agent.research.httpx.AsyncClient") as mock_class:
        mock_class.return_value.__aenter__ = AsyncMock(return_value=client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
        return await research._fetch_html("https://org.example")


def test_find_subpages_skips_a_malformed_link():
    """urljoin raises on it; one bad link must not fail the whole research."""
    from agent.research import _find_subpages

    html = '<a href="http://[oops/about">x</a><a href="/about">About</a>'
    assert _find_subpages(html, "https://test.org") == ["https://test.org/about"]


def test_find_subpages_caps_number_of_pages():
    from agent.research import MAX_SUBPAGES, _find_subpages

    html = "".join(f'<a href="/programs/{i}">p</a>' for i in range(10))
    assert len(_find_subpages(html, "https://test.org")) == MAX_SUBPAGES


async def test_scrape_site_reads_homepage_and_subpages():
    pages = {
        "https://test.org": _response(HOME_HTML),
        "https://test.org/about-us": _response("<p>Founded in 2010.</p>"),
        "https://test.org/programs/": _response("<p>Free tutoring program.</p>"),
    }
    text, urls = await _scrape("test.org", _fake_client(lambda url, **_: pages[url]))

    assert urls == ["https://test.org", "https://test.org/about-us", "https://test.org/programs/"]
    assert "Welcome to Test Org." in text
    assert "[Page: https://test.org/programs/]" in text and "Free tutoring program." in text


async def test_scrape_site_skips_failed_subpages():
    def get(url, **_):
        if url == "https://test.org":
            return _response(HOME_HTML)
        raise Exception("timeout")

    text, urls = await _scrape("https://test.org", _fake_client(get))
    assert urls == ["https://test.org"]
    assert "Welcome to Test Org." in text


async def test_research_org_does_not_cite_a_homepage_it_could_not_read():
    """A blocked or failed homepage was never read, so it must not become a citable source."""
    from agent.research import research_org

    with (
        patch("agent.research.scrape_site", new=AsyncMock(return_value=("Error: blocked", []))),
        patch("agent.research.search_web", new=AsyncMock(return_value=[])),
    ):
        result = await research_org("Test Org", "http://169.254.169.254/")
    assert result["sources"] == []


async def test_search_web_reads_the_tavily_key_when_it_searches():
    """A key loaded from .env after agent.research was imported must still be used."""
    from agent.research import search_web

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-late"}),
        patch("agent.research.AsyncTavilyClient") as client_class,
    ):
        client_class.return_value.search = AsyncMock(return_value={"results": []})
        await search_web("q")
    client_class.assert_called_once_with(api_key="tvly-late")
