"""Provider web-citation handling for web-grounded panellists.

Web-search models (Perplexity sonar, and anything routed through OpenRouter
with web access) write bodies that cite sources as bracket markers — `[3]`,
`[14]` — which index into citation metadata carried OUTSIDE the message
content. Issue #61: fanout persisted only `message.content`, so the URL list
never reached the capsule extractor, the synthesiser, or the saved
artifacts, and research capsules ended up with `sources_cited: ["[3]"]`.

Three cooperating pieces:

- `harvest(raw)` — collect citation metadata from a raw response dict.
  Checks the three shapes seen in the wild (probed 2026-06-12 against
  `openrouter/perplexity/sonar-pro`):
  1. top-level `search_results` (Perplexity native, `{title, url, date}`)
  2. top-level `citations` (Perplexity classic, URL strings)
  3. `choices[0].message.annotations` (OpenAI/OpenRouter `url_citation`
     shape — the ONLY one that survives the OpenRouter route; the top-level
     fields arrive null there)
  List order matches the body's 1-indexed `[n]` markers on every shape.
  Never raises; returns [] when nothing usable is present.

- `append_sources_footer(body, sources)` — make the body self-contained by
  appending a numbered `Sources:` footer behind a stable delimiter. Fanout
  calls this once per panellist, so every downstream consumer (extractor,
  synth, viewer, refine rounds) reads the same enriched body.

- `resolve_marker_sources(cited, body)` — deterministic backstop for the
  capsule pass: rewrite bare-marker `sources_cited` entries to the matching
  footer line. The extractor prompt asks for verbatim copying only; cheap
  extractor models are unreliable at marker-to-URL lookups, so Python owns
  the resolution.

Known gap: the streaming path rebuilds responses via
`litellm.stream_chunk_builder`, which does not carry annotation metadata
through, so streamed web panellists get no footer. Harvest just returns []
there — same behaviour as before this module existed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Stable, structurally detectable delimiter. `split_sources_footer` keys on
# it so the capsule pass can trim the prose without risking the footer, and
# `resolve_marker_sources` only reads numbered lines after it.
FOOTER_DELIM = "\n\n---\nSources:\n"

# Defensive cap — Perplexity rarely returns more than ~20 sources, but an
# annotation flood from a misbehaving provider shouldn't balloon the body.
MAX_FOOTER_SOURCES = 30

_TITLE_MAX_CHARS = 120
_LINE_MAX_CHARS = 400


class SourceRef(NamedTuple):
    url: str
    title: str | None = None


def _clean_url(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")) or any(c.isspace() for c in url):
        return None
    return url


def _clean_title(title: Any) -> str | None:
    # Titles are untrusted web text headed into downstream LLM prompts:
    # collapse whitespace/newlines and cap the length.
    if not isinstance(title, str):
        return None
    title = " ".join(title.split())
    if not title:
        return None
    return title[:_TITLE_MAX_CHARS]


def _from_search_results(items: Any) -> list[SourceRef]:
    refs = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _clean_url(item.get("url"))
            if url:
                refs.append(SourceRef(url=url, title=_clean_title(item.get("title"))))
    return refs


def _from_citations(items: Any) -> list[SourceRef]:
    refs = []
    if isinstance(items, list):
        for item in items:
            url = _clean_url(item)
            if url:
                refs.append(SourceRef(url=url))
    return refs


def _from_annotations(data: dict[str, Any]) -> list[SourceRef]:
    try:
        annotations = data["choices"][0]["message"]["annotations"]
    except (KeyError, IndexError, TypeError):
        return []
    refs = []
    if isinstance(annotations, list):
        for item in annotations:
            if not isinstance(item, dict):
                continue
            cite = item.get("url_citation")
            if not isinstance(cite, dict):
                continue
            url = _clean_url(cite.get("url"))
            if url:
                refs.append(SourceRef(url=url, title=_clean_title(cite.get("title"))))
    return refs


def harvest(raw: Any) -> list[SourceRef]:
    """Collect provider citation metadata from a raw response dict.

    Takes the `model_dump()` dict (not the response object) so LiteLLM's
    Pydantic models can't silently strip provider-extra fields between
    versions. Never raises — citation harvesting is best-effort enrichment
    and must not be able to fail a panellist that already answered.
    """
    if not isinstance(raw, dict):
        return []
    try:
        for refs in (
            _from_search_results(raw.get("search_results")),
            _from_citations(raw.get("citations")),
            _from_annotations(raw),
        ):
            if refs:
                return refs
    except Exception as e:  # noqa: BLE001 — enrichment must never fail the call
        logger.warning("citation harvest failed: %s", e)
    return []


# A line that reads as an entry in a provider-inlined reference list:
# leading number (bracketed or dotted) followed by text carrying a URL.
# Deliberately permissive — a false match only skips appending the footer,
# and only when every harvested URL is already present in the body too.
_NUMBERED_SOURCE_LINE_RE = re.compile(r"^\s*\[?\d{1,3}[\].:]?\s+\S*.*https?://", re.M)

# Any [n] marker in prose — used only for drift observability.
_BODY_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def append_sources_footer(body: str, sources: list[SourceRef]) -> str:
    """Append a numbered `Sources:` footer so the body resolves its own
    `[n]` markers.

    Numbering is the 1-indexed list position — the same indexing the
    provider's markers use — so entries must not be deduplicated or
    reordered here. Skipped when there are no sources, the body is empty,
    or the provider already inlined its own reference list (every source
    URL present AND a numbered source-list line exists — the structural
    check keeps URLs merely quoted in prose from suppressing the footer).
    """
    if not body.strip() or not sources:
        return body
    if all(s.url in body for s in sources) and _NUMBERED_SOURCE_LINE_RE.search(body):
        return body
    markers = [int(m) for m in _BODY_MARKER_RE.findall(body)]
    if markers and max(markers) > len(sources):
        # The list-position-equals-marker-number contract is provider
        # behaviour, not a guarantee; surface drift instead of hiding it.
        logger.debug("body cites marker [%d] but only %d sources were harvested", max(markers), len(sources))
    capped = sources[:MAX_FOOTER_SOURCES]
    lines = [f"[{i}] {s.title} - {s.url}" if s.title else f"[{i}] {s.url}" for i, s in enumerate(capped, 1)]
    footer = FOOTER_DELIM + "\n".join(lines)
    elided = len(sources) - len(capped)
    if elided:
        footer += f"\n[... {elided} more sources not listed ...]"
    return body.rstrip() + footer + "\n"


def split_sources_footer(body: str) -> tuple[str, str]:
    """Split a body into (prose, footer). The footer includes its delimiter
    so `prose + footer == body`; a body without a footer returns (body, "").

    The suffix after the delimiter must contain at least one numbered
    source line — prose that organically produces the delimiter text but
    no source list is not a footer, and treating it as one would feed
    bogus lines to `resolve_marker_sources` and exempt an arbitrary tail
    from capsule trimming.
    """
    idx = body.rfind(FOOTER_DELIM)
    if idx == -1:
        return body, ""
    footer = body[idx:]
    if not any(_FOOTER_LINE_RE.match(line.strip()) for line in footer.splitlines()):
        return body, ""
    return body[:idx], footer


# A sources_cited entry that is nothing but a marker: "[3]", "3", "[14]".
# Bare digits are intentional — extractors sometimes strip the brackets.
# The 3-digit cap keeps year-like values ("2023") out of the rewrite.
_MARKER_RE = re.compile(r"^\[?(\d{1,3})\]?$")
# A footer line: "[3] Title - https://..." — anchored so prose that merely
# contains brackets (e.g. `arr[3]`) can never enter the lookup table.
_FOOTER_LINE_RE = re.compile(r"^\[(\d{1,3})\]\s+(\S.*)$")


def resolve_marker_sources(cited: list[str], body: str) -> list[str]:
    """Rewrite bare-marker entries against the body's Sources footer.

    Only lines inside the footer block feed the lookup table, and only
    entries that are a lone `[n]`/`n` marker are rewritten — anything else
    (real URLs, paper titles, unresolvable markers) passes through
    unchanged in its original position.
    """
    _, footer = split_sources_footer(body)
    if not footer:
        return cited
    table: dict[str, str] = {}
    for line in footer.splitlines():
        m = _FOOTER_LINE_RE.match(line.strip())
        if m:
            table[m.group(1)] = m.group(2).strip()[:_LINE_MAX_CHARS]
    if not table:
        return cited
    out: list[str] = []
    for entry in cited:
        m = _MARKER_RE.match(entry.strip()) if isinstance(entry, str) else None
        out.append(table[m.group(1)] if m and m.group(1) in table else entry)
    return out
