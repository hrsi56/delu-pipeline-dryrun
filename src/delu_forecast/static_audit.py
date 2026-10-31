"""Dependency-closure scanner for the static Pages export (CP-3 item 3).

Item 3 requires the page to perform **zero runtime calls** to the Space or any
external service. That is a property of the document, so it is checked by reading
the document rather than by reasoning about how it was built: every position that
can cause the browser to fetch something is enumerated, and any absolute or
protocol-relative URL in one of them is a finding.

Hyperlinks are deliberately *not* findings. §9.2 requires the page to link the
Space and the public MLflow UI beneath the report; an `<a href>` is followed only
when a human clicks it. Anything that would fetch without a click -- a script,
a stylesheet, an image, a font, an iframe, a preconnect hint, a link `ping`, a
`fetch`/`XHR`/`WebSocket`/`sendBeacon` call -- is a finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Attributes whose value the browser fetches with no user action.
FETCHING_ATTRIBUTES: tuple[str, ...] = (
    "src", "srcset", "poster", "data", "codebase", "background", "ping", "formaction",
)

#: `<link rel=...>` values that fetch. `rel=alternate`/`canonical`/`author` do not.
FETCHING_LINK_RELS: tuple[str, ...] = (
    "stylesheet", "preload", "modulepreload", "prefetch", "preconnect", "dns-prefetch",
    "icon", "shortcut icon", "apple-touch-icon", "manifest", "prerender",
)

#: Script-initiated network APIs.
NETWORK_APIS: tuple[str, ...] = (
    "fetch(", "XMLHttpRequest", "WebSocket(", "EventSource(", "navigator.sendBeacon",
    "importScripts(", "import(", "navigator.serviceWorker",
)

_ABSOLUTE = re.compile(r"^\s*(?:[a-zA-Z][a-zA-Z0-9+.-]*:)?//")
_TAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9-]*)((?:\s+[^<>]*?)?)/?>", re.DOTALL)
_ATTR = re.compile(r"""([a-zA-Z_:@.\-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>`]+))""", re.DOTALL)
_CSS_URL = re.compile(r"url\(\s*['\"]?([^)'\"]+)['\"]?\s*\)", re.IGNORECASE)
_CSS_IMPORT = re.compile(r"@import\s+(?:url\()?\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def _is_external(value: str) -> bool:
    """Absolute or protocol-relative. `data:` and `#fragment` are local."""
    if value.startswith(("data:", "#", "?")):
        return False
    return bool(_ABSOLUTE.match(value))


def audit_html(document: str) -> list[Finding]:
    """Every way this document could reach the network without a click."""
    findings: list[Finding] = []

    for match in _TAG.finditer(document):
        tag = match.group(1).lower()
        attributes = {
            name.lower(): (double or single or bare or "")
            for name, double, single, bare in _ATTR.findall(match.group(2) or "")
        }
        if tag in {"iframe", "frame", "embed", "object", "applet", "portal"}:
            findings.append(Finding(f"<{tag}>", "embedded browsing context"))
        for attribute in FETCHING_ATTRIBUTES:
            value = attributes.get(attribute)
            if value and any(_is_external(part.strip()) for part in value.split(",")):
                findings.append(Finding(f"<{tag} {attribute}>", value[:120]))
        if tag == "link":
            rel = attributes.get("rel", "").lower().strip()
            href = attributes.get("href", "")
            if rel in FETCHING_LINK_RELS and href and _is_external(href):
                findings.append(Finding(f"<link rel={rel}>", href[:120]))
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            findings.append(Finding("<meta http-equiv=refresh>", attributes.get("content", "")[:120]))

    for pattern, kind in ((_CSS_URL, "css url()"), (_CSS_IMPORT, "css @import")):
        for match in pattern.finditer(document):
            if _is_external(match.group(1)):
                findings.append(Finding(kind, match.group(1)[:120]))

    for api in NETWORK_APIS:
        if api in document:
            findings.append(Finding("script network API", api))

    return findings


__all__ = ["FETCHING_ATTRIBUTES", "FETCHING_LINK_RELS", "Finding", "NETWORK_APIS", "audit_html"]
