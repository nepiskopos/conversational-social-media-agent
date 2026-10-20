"""
Website scraping and web search for org research.

Owns: fetching and parsing org websites (homepage plus About/Programs-style
pages, with an SSRF guard on every request and redirect), searching the web
via Tavily, and combining results into a structured research dict. The
searches, the website and its subpages are fetched in parallel. Does not own
storing research (memory.py), the org brief or tool dispatch (tools.py), or
the conversation loop.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
import unicodedata
import zlib
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tavily import AsyncTavilyClient

from .logs import describe, safe

logger = logging.getLogger(__name__)

MAX_REDIRECTS: int = 5
# Seconds to wait for one page (redirects included) or search; a stalled website must not
# hold up the reply.
FETCH_TIMEOUT_S: float = 15.0
# Bytes of a page's HTML read at most; only SITE_PAGE_CHARS of its text are kept anyway, and
# a huge or endless page must not fill the server's memory.
MAX_PAGE_BYTES: int = 2_000_000
# A website host: dot-separated labels ending in a top-level domain of letters (punycode too).
WEBSITE_HOST_PATTERN: re.Pattern[str] = re.compile(
    r"(?:[a-z0-9-]+\.)+(?:[a-z]{2,63}|xn--[a-z0-9-]+)"
)
# Characters around a typed website that aren't part of it: "(myorg.org)." or "myorg.org,".
WEBSITE_TRIM: str = ".,;:!?()[]<>\"'"
# Content-Encodings Mackie unpacks itself (see _read_page); it asks for those in ACCEPT_ENCODING.
ACCEPTED_ENCODINGS: dict[str, Callable[[], Any]] = {
    "identity": lambda: _Unchanged(),
    "gzip": lambda: zlib.decompressobj(zlib.MAX_WBITS | 16),
    "x-gzip": lambda: zlib.decompressobj(zlib.MAX_WBITS | 16),
    "deflate": lambda: _Deflate(),
}
ACCEPT_ENCODING: str = "identity, gzip, deflate"
# Compressions Mackie doesn't ask for or unpack; a page sent in one is skipped, as is one
# compressed twice ("gzip, gzip").
REFUSED_ENCODINGS: frozenset[str] = frozenset({"br", "zstd", "compress", "x-compress"})
ZLIB_METHOD_DEFLATE: int = 8
SITE_PAGE_CHARS: int = 4000  # per page when several pages of one site are combined
MAX_SUBPAGES: int = 3
# URL path fragments of the pages that usually describe what an org does.
SUBPAGE_KEYWORDS: tuple[str, ...] = (
    "about",
    "program",
    "service",
    "what-we-do",
    "our-work",
    "mission",
    "impact",
    "event",
    "get-involved",
)

# Hostnames that must never be fetched — cloud metadata endpoints and localhost.
# IP-based checks below cover the rest of the private/loopback space.
_BLOCKED_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata.google.internal",  # GCP's metadata hostname (its address is link-local)
    }
)


# IPv6 ranges that carry an IPv4 address in their last 32 bits: NAT64 gateways
# (RFC 6052) translate 64:ff9b::a9fe:a9fe to 169.254.169.254, and the deprecated
# IPv4-compatible form ::a.b.c.d embeds one directly. Judged by the IPv4 address.
_IPV4_EMBEDDING_NETWORKS: tuple[ipaddress.IPv6Network, ...] = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("::/96"),
    ipaddress.IPv6Network("::ffff:0:0:0/96"),  # IPv4-translated (RFC 2765), for SIIT
)
# Local-use NAT64 prefix (RFC 8215): operators put it in front of their own networks.
_LOCAL_NAT64_NETWORK: ipaddress.IPv6Network = ipaddress.IPv6Network("64:ff9b:1::/48")
_IPV4_BITS: int = 0xFFFFFFFF


def _is_internal_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    True for any address that is not a public unicast address.

    "Not global" covers loopback, private, link-local, reserved, unspecified
    (0.0.0.0, ::) and carrier-grade NAT (100.64.0.0/10, home of Alibaba
    Cloud's metadata endpoint 100.100.100.200), which is_private misses.
    IPv6 addresses that carry an IPv4 address (IPv4-mapped ::ffff:127.0.0.1,
    NAT64 64:ff9b::7f00:1, IPv4-compatible ::127.0.0.1) are checked as that
    IPv4 address; Python counts NAT64 as global, which let a NAT64 gateway
    reach internal IPv4 hosts.
    """
    if isinstance(addr, ipaddress.IPv6Address):
        if addr in _LOCAL_NAT64_NETWORK:
            return True
        if addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        elif any(addr in network for network in _IPV4_EMBEDDING_NETWORKS):
            addr = ipaddress.IPv4Address(int(addr) & _IPV4_BITS)
    return not addr.is_global or addr.is_multicast


def with_scheme(url: str) -> str:
    """Add "https://" to a URL that has no http(s) scheme; the check ignores case."""
    return url if re.match(r"https?://", url, re.IGNORECASE) else "https://" + url


def website_url(text: str) -> str:
    """
    The website a user typed, as an http(s) URL, or "" if it doesn't look like one.

    "myorg.org" becomes "https://myorg.org", "(myorg.org)." too, and a name
    in another script is checked in its ASCII (IDNA) form. Text such as
    "Inc." or "our site", a login in the URL, or a port that isn't a number
    gives "", so it is never stored as the org's website.
    """
    text = text.strip().strip(WEBSITE_TRIM)
    if not text or any(char.isspace() for char in text):
        return ""
    url = with_scheme(text)
    parsed = urlparse(url)
    try:
        parsed.port  # noqa: B018 - raises ValueError for a port that isn't a valid number
        host = (parsed.hostname or "").encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        return ""
    if parsed.username is not None or not WEBSITE_HOST_PATTERN.fullmatch(host):
        return ""
    return url


def _is_safe_url(url: str) -> bool:
    """
    Return True only if the URL is safe to fetch (public internet, HTTPS or HTTP).

    Blocks:
    - Non-HTTP(S) schemes (file://, ftp://, etc.)
    - Known metadata hostnames, localhost and *.localhost
    - IP addresses that are not public (see _is_internal_address)
    - Non-standard IP representations (hex "0x7f000001", decimal "2130706433",
      octal "0177.0.0.1", short "127.1") that ipaddress.ip_address() rejects
      but some resolvers accept
    - Hosts disguised with percent-encoding or fullwidth digits, which are
      decoded and normalised before any of the checks above

    Domain names are checked again after DNS lookup by _resolves_to_internal.

    Args:
        url: Fully-qualified URL (https:// prefix already applied).

    Returns:
        True if the URL is safe to fetch, False otherwise.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    # Decode before judging: "%31%32%37.0.0.1" and fullwidth "１２７.０.０.１" both
    # become 127.0.0.1 further down the stack (IDNA maps fullwidth digits to ASCII).
    host = unicodedata.normalize("NFKC", unquote(parsed.hostname or "")).lower().rstrip(".")
    if not host:
        return False

    if host in _BLOCKED_HOSTS or host.endswith(".localhost"):
        return False

    try:
        return not _is_internal_address(ipaddress.ip_address(host))
    except ValueError:
        pass  # not a standard IP address

    # Hex ("0x7f000001"), decimal ("2130706433"), octal ("0177.0.0.1") and short
    # ("127.1") forms are not standard IP addresses, but some resolvers read them
    # as 127.0.0.1. No real domain name looks like this, so refuse them all.
    return not (host.startswith("0x") or re.fullmatch(r"[0-9.]+", host))


class BlockedURLError(ValueError):
    """The URL (or a redirect target) points at a private, loopback or metadata address."""


async def _resolve_host(host: str) -> list[str]:
    """IP addresses a hostname resolves to; empty when it does not resolve."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return [str(info[4][0]) for info in infos]


async def _resolves_to_internal(url: str) -> bool:
    """
    True if the URL's hostname resolves to any internal address.

    Closes the gap _is_safe_url leaves: it only sees the name, so a public
    domain whose DNS points at 127.0.0.1 or a cloud metadata IP passed. A name
    that does not resolve is left to the HTTP request to fail.
    """
    addresses = await _resolve_host(urlparse(url).hostname or "")
    # IPv6 link-local results carry a zone suffix ("fe80::1%eth0") that ip_address() rejects.
    return any(_is_internal_address(ipaddress.ip_address(a.split("%")[0])) for a in addresses)


async def _fetch_html(url: str) -> tuple[str, str]:
    """
    GET a page safely and return (final_url, html).

    Redirects are followed by hand, up to MAX_REDIRECTS, and every hop is
    re-checked with _is_safe_url and a DNS lookup (_resolves_to_internal) —
    httpx's follow_redirects=True would fetch unchecked targets. Remaining gap:
    httpx resolves the name again when it connects, so a DNS server that
    answers differently the second time (DNS rebinding) is not caught.

    The whole fetch, lookups and redirects included, gives up after
    FETCH_TIMEOUT_S: httpx's own timeout bounds each wait for data, so a server
    sending a byte at a time could otherwise hold it open for good. At most
    MAX_PAGE_BYTES of the page are read.

    Raises:
        BlockedURLError: the URL or a redirect target is not safe to fetch.
        ValueError:      too many redirects, or a malformed redirect target.
        TimeoutError:    the page took longer than FETCH_TIMEOUT_S.
        httpx.HTTPError: network failure or an HTTP error status.
    """
    async with asyncio.timeout(FETCH_TIMEOUT_S), httpx.AsyncClient() as client:
        for _ in range(MAX_REDIRECTS + 1):
            if not _is_safe_url(url) or await _resolves_to_internal(url):
                raise BlockedURLError(f"URL '{url}' is not allowed (not a public http(s) address).")
            async with client.stream(
                "GET",
                url,
                timeout=FETCH_TIMEOUT_S,
                follow_redirects=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; MackieBot/1.0)",
                    "Accept-Encoding": ACCEPT_ENCODING,
                },
            ) as response:
                if not response.is_redirect:
                    response.raise_for_status()
                    return url, await _read_page(response)
                location = response.headers["location"]
            url = urljoin(url, location)
    raise ValueError("too many redirects")


async def _read_page(response: httpx.Response) -> str:
    """
    A streamed response's body as text, reading at most MAX_PAGE_BYTES of it, unpacked.

    The body is unpacked here, never past MAX_PAGE_BYTES, rather than by
    httpx, which unpacks each chunk whole: 2 KB of nested gzip from a hostile
    site became 1 GB in memory. One layer of gzip or deflate is unpacked
    (ACCEPTED_ENCODINGS); an unknown value ("none") is read as plain text.

    Raises:
        ValueError: the page is sent in a compression Mackie doesn't unpack, or several.
    """
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding in REFUSED_ENCODINGS or "," in encoding:
        raise ValueError(f"page sent with Content-Encoding {encoding[:40]!r}")
    # Anything else unknown ("none", "utf-8": misconfigured servers send these) is plain text.
    unpacker = ACCEPTED_ENCODINGS.get(encoding, _Unchanged)()
    body = bytearray()  # grows in place: bytes += would copy the page on every chunk
    received = 0
    async for chunk in response.aiter_raw():
        received += len(chunk)
        body += unpacker.decompress(chunk, MAX_PAGE_BYTES - len(body))
        # Bytes after the end of a compressed stream pile up unread inside the unpacker,
        # so the bytes received are capped too, and reading stops at the stream's end.
        if len(body) >= MAX_PAGE_BYTES or received >= MAX_PAGE_BYTES or unpacker.eof:
            break
    return bytes(body).decode(response.encoding or "utf-8", errors="replace")


class _Unchanged:
    """The unpacker for an uncompressed page: its bytes, up to the limit given."""

    eof: bool = False  # an uncompressed page ends when the server stops sending

    def decompress(self, data: bytes, max_length: int) -> bytes:
        return data[:max_length]


class _Deflate:
    """
    The unpacker for "deflate": zlib-wrapped, as the standard says, or raw, as some servers send.

    Which one is told by the first byte (a zlib header's low four bits are 8),
    as httpx does.
    """

    def __init__(self) -> None:
        self._unpacker: Any = None

    @property
    def eof(self) -> bool:
        """Whether the compressed stream has ended."""
        return bool(self._unpacker and self._unpacker.eof)

    def decompress(self, data: bytes, max_length: int) -> bytes:
        if self._unpacker is None:
            wrapped = bool(data) and data[0] & 0x0F == ZLIB_METHOD_DEFLATE
            self._unpacker = zlib.decompressobj(zlib.MAX_WBITS if wrapped else -zlib.MAX_WBITS)
        return self._unpacker.decompress(data, max_length)


def _page_text(html: str) -> str:
    """Extract readable text from HTML, dropping scripts, styles and site chrome."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _find_subpages(html: str, base_url: str) -> list[str]:
    """
    Return up to MAX_SUBPAGES same-site links that likely describe the org.

    Nav and footer links are included on purpose — that is where "About" and
    "Programs" usually live. Matching is on the URL path only. A malformed
    link (such as "http://[oops") is skipped.
    """
    base = urlparse(base_url)
    site = base.hostname.removeprefix("www.") if base.hostname else ""
    found: list[str] = []
    for link in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        try:
            url = urljoin(base_url, link["href"]).split("#")[0]
            parsed = urlparse(url)
            host = (parsed.hostname or "").removeprefix("www.")
        except ValueError:
            continue
        path = parsed.path.lower()
        if (
            host == site
            and parsed.scheme in ("http", "https")
            and path.strip("/")
            and not path.endswith((".pdf", ".jpg", ".png", ".zip"))
            and any(keyword in path for keyword in SUBPAGE_KEYWORDS)
            and url.rstrip("/") != base_url.rstrip("/")
            and url not in found
        ):
            found.append(url)
        if len(found) == MAX_SUBPAGES:
            break
    return found


async def scrape_site(url: str) -> tuple[str, list[str]]:
    """
    Read an org's homepage plus its About/Programs/Events-style pages.

    Programs and mission statements usually live one click from the homepage,
    so up to MAX_SUBPAGES same-site pages linked from it are fetched in
    parallel. Each page's text is labelled with its URL so it can be cited.

    Args:
        url: Website URL. An "https://" prefix is added if missing.

    Returns:
        (combined_text, page_urls). A homepage that can't be read gives ("", [])
        and a line in the log (an error's text is no research); failed subpages
        are skipped.
    """
    url = with_scheme(url)
    try:
        home_url, html = await _fetch_html(url)
    except BlockedURLError as e:
        # Worth an operator's eye: a page or a model asked Mackie to fetch an internal address.
        logger.warning("Refused to fetch %s: %s", safe(url), describe(e))
        return "", []
    except Exception as e:
        logger.warning("Could not fetch %s: %s", safe(url), describe(e))
        return "", []

    subpages = _find_subpages(html, home_url)
    sub_results = await asyncio.gather(*(_fetch_html(u) for u in subpages), return_exceptions=True)
    pages = [(home_url, html)] + [r for r in sub_results if not isinstance(r, BaseException)]
    text = "\n\n".join(
        f"[Page: {page_url}]\n{_page_text(page_html)[:SITE_PAGE_CHARS]}"
        for page_url, page_html in pages
    )
    return text, [page_url for page_url, _ in pages]


async def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web via Tavily and return structured results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet. On error, returns
        an empty list, and a line in the log.
    """
    try:
        # Read per call, not at import, so it works whenever the caller loads .env.
        client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        # Tavily's own timeout bounds each wait for data, not the whole search.
        async with asyncio.timeout(FETCH_TIMEOUT_S):
            response = await client.search(query, max_results=max_results, timeout=FETCH_TIMEOUT_S)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in response.get("results", [])
            if r.get("url")
        ]
    except Exception as e:
        logger.warning("Web search failed: %s", describe(e))
        return []


async def research_org(org_name: str, org_url: str = "") -> dict[str, Any]:
    """
    Research an org by scraping its website and running web searches in parallel.

    Args:
        org_name: Human-readable name used as the search query base.
        org_url:  Optional website URL. If provided, the homepage and its
                  About/Programs-style pages are scraped (see scrape_site).

    Returns:
        Dict with keys: org_name, website_content, social_presence, news,
        similar_orgs, sources (dicts with type and url, and a title for search results).
    """
    social_coro = search_web(
        f'"{org_name}" site:instagram.com OR site:facebook.com OR site:linkedin.com', 4
    )
    this_year = date.today().year
    news_coro = search_web(f'"{org_name}" nonprofit {this_year - 1} OR {this_year}', 5)
    similar_coro = search_web(f'nonprofit similar to "{org_name}" social media content strategy', 3)

    if org_url:
        (
            (website_content, page_urls),
            social_results,
            news_results,
            similar_results,
        ) = await asyncio.gather(scrape_site(org_url), social_coro, news_coro, similar_coro)
    else:
        website_content, page_urls = "", []
        social_results, news_results, similar_results = await asyncio.gather(
            social_coro, news_coro, similar_coro
        )

    # Only pages that were read: a homepage that failed or was blocked is not a source.
    sources = [{"type": "website", "url": page_url} for page_url in page_urls] + [
        {"type": kind, "url": r["url"], "title": r["title"]}
        for kind, results in (("social", social_results), ("news", news_results))
        for r in results
    ]
    return {
        "org_name": org_name,
        "website_content": website_content,
        "social_presence": social_results,
        "news": news_results,
        "similar_orgs": similar_results,
        "sources": sources,
    }
