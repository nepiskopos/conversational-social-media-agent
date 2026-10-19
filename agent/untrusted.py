"""
Trust-boundary tags for text Mackie did not write: scraped pages, search
snippets, and the org brief and conversation summary derived from them.

Owns: wrapping such text in a named tag that the prompt marks as data (the
Security Policy for <external_content>; a note beside the brief and the
summary), and neutralising any copy of that tag inside the text so
a page containing "</external_content>" cannot close the wrapper early and pass
the rest of its text off as instructions. Does not own deciding what is
untrusted; callers choose what to wrap.
"""

from __future__ import annotations

import html
import re


def wrap_untrusted(tag: str, text: str, **attributes: str) -> str:
    """
    Enclose text in <tag attr="...">…</tag>, so that it cannot break out.

    Any opening or closing copy of the tag inside the text (in any case, with
    any spacing) has its "<" escaped to "&lt;", and attribute values are
    HTML-escaped, so a quote in a URL cannot end the attribute.

    Args:
        tag:        Tag name, e.g. "external_content".
        text:       The untrusted text.
        attributes: Tag attributes, e.g. source="web_search", url="https://…".

    Returns:
        The wrapped text, with the tags on their own lines.
    """
    inner = re.sub(rf"<(\s*/?\s*{re.escape(tag)})", r"&lt;\1", text, flags=re.IGNORECASE)
    attrs = "".join(f' {name}="{html.escape(value)}"' for name, value in attributes.items())
    return f"<{tag}{attrs}>\n{inner}\n</{tag}>"
