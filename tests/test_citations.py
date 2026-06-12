"""Tests for consult/citations.py — provider web-citation harvest, the
Sources footer, and bare-marker resolution (issue #61)."""

from __future__ import annotations

from consult.citations import (
    FOOTER_DELIM,
    MAX_FOOTER_SOURCES,
    SourceRef,
    append_sources_footer,
    harvest,
    resolve_marker_sources,
    split_sources_footer,
)

# ---------------------------------------------------------------- harvest


def _openrouter_raw(annotations):
    return {"choices": [{"message": {"content": "body [1][2]", "annotations": annotations}}]}


def test_harvest_reads_openrouter_url_citation_annotations():
    """The shape that survives the OpenRouter route (probed 2026-06-12
    against openrouter/perplexity/sonar-pro): message.annotations with
    url_citation entries; top-level citations/search_results arrive null."""
    raw = _openrouter_raw(
        [
            {"type": "url_citation", "url_citation": {"url": "https://a.example/x", "title": "A"}},
            {"type": "url_citation", "url_citation": {"url": "https://b.example/y", "title": "B"}},
        ]
    )
    raw["citations"] = None
    raw["search_results"] = None
    assert harvest(raw) == [
        SourceRef(url="https://a.example/x", title="A"),
        SourceRef(url="https://b.example/y", title="B"),
    ]


def test_harvest_reads_top_level_citations():
    raw = {"citations": ["https://a.example/x", "https://b.example/y"]}
    assert harvest(raw) == [SourceRef(url="https://a.example/x"), SourceRef(url="https://b.example/y")]


def test_harvest_prefers_search_results_and_preserves_order():
    """search_results carries titles, so it wins over the bare-URL
    citations list; list order is the 1-indexed marker order and must
    never be reshuffled."""
    raw = {
        "search_results": [
            {"url": "https://b.example/second", "title": "Second"},
            {"url": "https://a.example/first", "title": "First"},
        ],
        "citations": ["https://other.example"],
    }
    assert harvest(raw) == [
        SourceRef(url="https://b.example/second", title="Second"),
        SourceRef(url="https://a.example/first", title="First"),
    ]


def test_harvest_tolerates_garbage_without_raising():
    assert harvest(None) == []
    assert harvest("not a dict") == []
    assert harvest({}) == []
    assert harvest({"choices": []}) == []
    assert harvest({"choices": [{"message": {}}]}) == []
    assert harvest(_openrouter_raw("not a list")) == []
    assert harvest(_openrouter_raw([{"type": "url_citation"}])) == []  # no url_citation dict
    assert harvest(_openrouter_raw([{"url_citation": {"title": "no url"}}])) == []
    assert harvest({"citations": [42, None, {}]}) == []


def test_harvest_rejects_non_http_urls_and_sanitises_titles():
    raw = {
        "search_results": [
            {"url": "ftp://nope.example", "title": "skipped"},
            {"url": "javascript:alert(1)", "title": "skipped"},
            {"url": "https://ok.example", "title": "  multi\nline\t title  "},
            {"url": "https://ws.example/with space", "title": "skipped"},
            {"url": "https://long.example", "title": "x" * 500},
        ]
    }
    refs = harvest(raw)
    assert [r.url for r in refs] == ["https://ok.example", "https://long.example"]
    assert refs[0].title == "multi line title"
    assert refs[1].title is not None and len(refs[1].title) == 120


# ------------------------------------------------------ append + split


def test_append_sources_footer_numbers_from_one():
    body = "Latest release is 3.14.6.[1][2]"
    out = append_sources_footer(
        body,
        [SourceRef(url="https://a.example", title="A"), SourceRef(url="https://b.example")],
    )
    assert out.startswith(body)
    assert FOOTER_DELIM in out
    assert "[1] A - https://a.example\n" in out
    assert "[2] https://b.example\n" in out


def test_append_sources_footer_noop_without_sources_or_body():
    assert append_sources_footer("body", []) == "body"
    assert append_sources_footer("", [SourceRef(url="https://a.example")]) == ""
    assert append_sources_footer("   \n", [SourceRef(url="https://a.example")]) == "   \n"


def test_append_sources_footer_skips_when_provider_inlined_reference_list():
    """A provider that inlines its own numbered reference list doesn't need
    a second one; but if even one URL is missing, the footer is still
    appended."""
    body = "Claim.[1][2]\n\nReferences:\n[1] https://a.example\n[2] https://b.example"
    refs = [SourceRef(url="https://a.example"), SourceRef(url="https://b.example")]
    assert append_sources_footer(body, refs) == body

    refs.append(SourceRef(url="https://c.example"))
    out = append_sources_footer(body, refs)
    assert "[3] https://c.example" in out


def test_append_sources_footer_appends_when_urls_only_quoted_in_prose():
    """URLs merely mentioned in prose are not a reference list — the [n]
    markers still need the footer to resolve. Substring presence alone
    must not suppress it."""
    body = "Compare https://a.example and https://b.example as discussed.[1][2]"
    refs = [SourceRef(url="https://a.example"), SourceRef(url="https://b.example")]
    out = append_sources_footer(body, refs)
    assert FOOTER_DELIM in out
    assert "[1] https://a.example" in out


def test_append_sources_footer_caps_and_notes_elision():
    refs = [SourceRef(url=f"https://s{i}.example") for i in range(MAX_FOOTER_SOURCES + 5)]
    out = append_sources_footer("body", refs)
    assert f"[{MAX_FOOTER_SOURCES}] https://s{MAX_FOOTER_SOURCES - 1}.example" in out
    assert f"[{MAX_FOOTER_SOURCES + 1}]" not in out
    assert "[... 5 more sources not listed ...]" in out


def test_split_sources_footer_roundtrip():
    body = "prose with [1] marker"
    out = append_sources_footer(body, [SourceRef(url="https://a.example", title="A")])
    prose, footer = split_sources_footer(out)
    assert prose + footer == out
    assert prose == body
    assert footer.startswith(FOOTER_DELIM)

    assert split_sources_footer("no footer here") == ("no footer here", "")


def test_split_sources_footer_rejects_organic_delimiter_without_source_lines():
    """Prose that organically produces the delimiter text but carries no
    numbered source lines after it is not a footer — splitting there would
    exempt an arbitrary tail from trimming and feed garbage to the
    resolver."""
    body = "Discussing footers." + FOOTER_DELIM + "are a way to list things, the essay continued."
    assert split_sources_footer(body) == (body, "")


# ------------------------------------------------------------- resolve


def _footered_body() -> str:
    return append_sources_footer(
        "Claim one.[1] Claim two.[3]",
        [
            SourceRef(url="https://a.example/one", title="One"),
            SourceRef(url="https://b.example/two"),
            SourceRef(url="https://c.example/three", title="Three"),
        ],
    )


def test_resolve_marker_sources_rewrites_bare_markers():
    body = _footered_body()
    out = resolve_marker_sources(["[1]", "[3]", "3"], body)
    assert out == [
        "One - https://a.example/one",
        "Three - https://c.example/three",
        "Three - https://c.example/three",
    ]


def test_resolve_marker_sources_passes_through_non_markers():
    body = _footered_body()
    cited = ["https://elsewhere.example", "Smith et al. 2024", "[99]", "[1], [2]"]
    assert resolve_marker_sources(cited, body) == cited


def test_resolve_marker_sources_leaves_year_like_values_alone():
    """Four-digit entries ("2023") must never be treated as markers — the
    3-digit cap in the marker regex is intentional."""
    body = _footered_body()
    assert resolve_marker_sources(["2023", "[2024]"], body) == ["2023", "[2024]"]


def test_resolve_marker_sources_without_footer_is_noop():
    cited = ["[1]", "[2]"]
    assert resolve_marker_sources(cited, "body that mentions [1] inline") == cited


def test_resolve_marker_sources_ignores_numbered_lines_outside_footer():
    """Prose that happens to contain `[2] something` lines must not feed the
    lookup table — only lines inside the footer block do."""
    body = (
        "[2] this is prose, not a source list\n\nreal claim [1]" + FOOTER_DELIM + "[1] https://real.example"
    )
    out = resolve_marker_sources(["[1]", "[2]"], body)
    assert out == ["https://real.example", "[2]"]
