"""HTML feed renderer.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from smoke_helpers import _make_run_dir

from consult import artifacts


def test_viewer_markdown_subset_renders_each_construct():
    """The viewer ships its own small markdown renderer (rather than pulling
    in a library) — pin every construct the synthesiser actually emits so a
    regex regression doesn't silently lose formatting in `feed.html`.
    """
    from consult.viewer import render_markdown

    out = render_markdown("# Heading 1\n## Heading 2")
    assert "<h1>Heading 1</h1>" in out
    assert "<h2>Heading 2</h2>" in out

    out = render_markdown("- one\n- two\n- three")
    assert out.count("<li>") == 3
    assert "<ul>" in out and "</ul>" in out

    out = render_markdown("1. first\n2. second")
    assert "<ol>" in out and out.count("<li>") == 2

    out = render_markdown("Some **bold** and *italic* with `code` and a [link](https://x.test).")
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out
    assert "<code>code</code>" in out
    assert '<a href="https://x.test">link</a>' in out

    out = render_markdown("```python\nprint(1)\n```")
    assert '<pre><code class="lang-python">print(1)</code></pre>' in out

    out = render_markdown("para one\n\npara two")
    assert out.count("<p>") == 2


def test_viewer_markdown_escapes_untrusted_html():
    """A panellist body that contained `<script>` must not become live HTML
    when threaded through synthesis — every line passes through html.escape
    before inline transforms run.
    """
    from consult.viewer import render_markdown

    out = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out

    out = render_markdown("- <img src=x onerror=alert(1)>")
    assert "<img" not in out
    assert "&lt;img" in out


def test_viewer_markdown_inline_code_protects_bold_markers():
    """Inline-code spans must be stashed before the bold regex runs;
    otherwise `` `**foo**` `` is wrongly rendered with a nested <strong>.
    """
    from consult.viewer import render_markdown

    out = render_markdown("Literal `**not bold**` here.")
    assert "<code>**not bold**</code>" in out
    assert "<strong>" not in out


def test_viewer_fmt_cost_handles_unknown_and_small_values():
    """`cost_usd=None` and `cost_known=False` must not crash the renderer."""
    from consult.viewer import _fmt_cost, _fmt_ms

    assert _fmt_cost(None) == "—"
    assert _fmt_cost(0) == "$0"
    assert _fmt_cost(0.0123) == "$0.0123"
    assert _fmt_cost(2.5).startswith("$2.5")
    # Unknown-pricing is flagged with a trailing `*` — mirrors ledger output.
    assert _fmt_cost(0.5, known=False).endswith("*")
    assert _fmt_ms(None) == "—"
    assert _fmt_ms(250) == "250ms"
    assert _fmt_ms(2500).endswith("s")


def test_viewer_render_run_panel_includes_core_sections(tmp_path, monkeypatch):
    """Panel-only runs (no synthesis) still render — manifest table, capsule,
    body. The 'panel' kind is derived from the absence of synthesis.md.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-1"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha",
                "model_id": "anthropic/claude-opus-4-7",
                "status": "OK",
                "capsule": {
                    "position": "supports A",
                    "recommendation": "ship A",
                    "key_points": ["fast", "cheap"],
                },
                "resource_uri": f"consult://runs/{rid}/responses/alpha",
                "body_path": "/x",
                "latency_ms": 4200,
                "cost_usd": 0.012,
                "cost_known": True,
                "confidence": 0.8,
            }
        ],
        bodies={"alpha": "alpha body text"},
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert out.name == "feed.html"
    assert "<!doctype html>" in text
    assert rid in text
    # Panel run: no synthesis section, no arbiter section, but core panellist
    # card is present with the capsule fields surfaced.
    assert "panel" in text.lower()
    assert "<h2>Synthesis</h2>" not in text
    assert "<h2>Arbiter rounds</h2>" not in text
    assert "alpha" in text
    assert "supports A" in text
    assert "ship A" in text
    assert "anthropic/claude-opus-4-7" in text
    # The status pill must use the OK tone.
    assert "pill-ok" in text


def test_viewer_render_run_consult_renders_synthesis_markdown(tmp_path, monkeypatch):
    """A consult run has synthesis.md — the viewer renders it through the
    markdown subset, not as a raw `<pre>` blob. Catches a regression where
    we'd accidentally drop the synthesis section.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-2"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            }
        ],
        synth="# Consensus\n\n- point one\n- point two",
        extras={"synthesiser": "gemini-pro"},
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "<h2>Synthesis</h2>" in text
    assert "<h1>Consensus</h1>" in text
    assert "<li>point one</li>" in text
    assert "consult" in text.lower()
    # `synthesiser` field surfaces in the header stats so the reader can see
    # which model produced the synthesis without opening the manifest.
    assert "gemini-pro" in text


def test_viewer_render_run_escapes_panellist_bodies(tmp_path, monkeypatch):
    """Untrusted panellist response text must be HTML-escaped before
    landing in `feed.html` — otherwise a body containing `<script>` would
    execute when the user opened the page in a browser.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-4"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            }
        ],
        bodies={"alpha": "<script>alert(1)</script>"},
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_viewer_render_run_renders_progress_timeline(tmp_path, monkeypatch):
    """`_progress.log` events become a chronological timeline section with
    deltas computed from the first event.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-5"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            }
        ],
        progress_lines=[
            {
                "ts": "2026-05-20T10:18:09.000000+00:00",
                "done": 0,
                "total": 0,
                "kind": "panellist_completed",
                "slug": "alpha",
                "status": "OK",
                "latency_ms": 1234,
            },
            {
                "ts": "2026-05-20T10:18:13.500000+00:00",
                "done": 1,
                "total": 1,
                "kind": "capsule_extracted",
                "slug": "alpha",
            },
        ],
    )
    out = viewer.render_run(rid)
    text = out.read_text()
    assert "<h2>Timeline</h2>" in text
    assert "panellist_completed" in text
    assert "capsule_extracted" in text
    # Relative offset: first event is +0.0s, second is +4.5s after it.
    assert "+0.0s" in text
    assert "+4.5s" in text


def test_viewer_render_run_raises_for_missing_run(tmp_path, monkeypatch):
    """Unknown run_id → FileNotFoundError, which the CLI surfaces as an
    exit-code-1 error rather than a stack trace.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        viewer.render_run("20990101-nope-1")


def test_viewer_render_run_raises_for_run_without_manifest(tmp_path, monkeypatch):
    """A run dir that exists but has no manifest can't be rendered — bail
    out with a clear message rather than producing a half-built page.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-no-manifest"
    (tmp_path / rid).mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json missing"):
        viewer.render_run(rid)


def test_viewer_render_run_surfaces_partial_and_cancelled_banners(tmp_path, monkeypatch):
    """Partial reason and the CANCELLED marker must surface visually so the
    reader doesn't mistake a half-finished run for a clean one.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-7"
    root = _make_run_dir(
        tmp_path,
        rid,
        entries=[],
        extras={"partial": True, "partial_reason": "cost cap exceeded"},
    )
    (root / "CANCELLED").touch()
    text = viewer.render_run(rid).read_text()
    assert "cost cap exceeded" in text
    assert "cancelled" in text.lower()


def test_viewer_cli_writes_path_to_stdout(tmp_path, monkeypatch, capsys):
    """`consult-view <run_id>` prints the absolute path so users can pipe
    it into `open(1)` or copy/paste it. No --open flag = no browser launch.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-cli"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            }
        ],
    )

    opened: list[str] = []
    monkeypatch.setattr(viewer.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(sys, "argv", ["consult-view", rid])
    viewer.cli()

    out = capsys.readouterr().out.strip()
    assert out.endswith("feed.html")
    assert Path(out).exists()
    assert opened == []  # --open not passed


def test_viewer_cli_open_flag_launches_browser(tmp_path, monkeypatch, capsys):
    """`--open` invokes webbrowser.open with the file:// URI of feed.html."""
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    rid = "20260520-fake-cli-open"
    _make_run_dir(
        tmp_path,
        rid,
        entries=[
            {
                "slug": "alpha",
                "model_id": "x/y",
                "status": "OK",
                "resource_uri": f"consult://runs/{rid}/responses/alpha",
                "body_path": "/x",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "cost_known": True,
            }
        ],
    )

    opened: list[str] = []
    monkeypatch.setattr(viewer.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(sys, "argv", ["consult-view", rid, "--open"])
    viewer.cli()

    assert len(opened) == 1
    assert opened[0].startswith("file://")
    assert opened[0].endswith("/feed.html")


def test_viewer_cli_missing_run_exits_with_message(tmp_path, monkeypatch, capsys):
    """An unknown run_id surfaces as a SystemExit (exit code != 0), not a
    raw traceback — keeps the CLI feeling like the rest of the unix tools.
    """
    from consult import viewer

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["consult-view", "20990101-no-such-run"])
    with pytest.raises(SystemExit) as ei:
        viewer.cli()
    # SystemExit carries the message in `code` when raised with a string.
    assert "Run not found" in str(ei.value) or "no-such-run" in str(ei.value)


def test_render_attachment_bare_string_renders_path_and_content(tmp_path):
    """Bare string attachment paths still work (backwards compat)."""
    from consult.attachments import render_attachment as _render_attachment

    f = tmp_path / "foo.py"
    f.write_text("print('hi')")
    out = _render_attachment(str(f))
    assert str(f) in out
    assert "print('hi')" in out
    assert "```" in out  # code-fenced


def test_render_attachment_labelled_renders_label_heading(tmp_path):
    """Labelled attachments render `## LABEL: path` headers so panellists
    can refer to sections by name."""
    from consult.attachments import render_attachment as _render_attachment

    f = tmp_path / "auth.py"
    f.write_text("def login(): pass")
    out = _render_attachment({"path": str(f), "label": "AUTH_MODULE", "kind": "source"})
    assert "## AUTH_MODULE" in out
    assert str(f) in out
    assert "def login()" in out


def test_render_attachment_kind_hints_fence_language(tmp_path):
    """`kind: "diff"` produces a ```diff fence so the panellist sees the
    syntax-highlighting hint."""
    from consult.attachments import render_attachment as _render_attachment

    f = tmp_path / "patch.diff"
    f.write_text("--- a/foo\n+++ b/foo\n@@ +1\n+hello")
    out = _render_attachment({"path": str(f), "kind": "diff"})
    assert "```diff" in out


def test_render_attachment_missing_file_renders_error_not_crash(tmp_path):
    """A missing file produces an inline error marker — the rest of the
    panel still runs."""
    from consult.attachments import render_attachment as _render_attachment

    out = _render_attachment(str(tmp_path / "nonexistent.py"))
    assert "ERROR" in out
    assert "nonexistent.py" in out


def test_render_attachment_malformed_dict_surfaces_error():
    """A dict without `path` or `source` keys surfaces an error rather
    than crashing the whole tool call."""
    from consult.attachments import render_attachment as _render_attachment

    out = _render_attachment({"foo": "bar"})
    assert "ERROR" in out


def test_inline_attachments_renders_each_item():
    """`_inline_attachments` chains render output and prepends the
    --- ATTACHMENTS --- divider."""
    from consult.attachments import inline_attachments as _inline_attachments

    out = _inline_attachments("PROMPT", [])
    # Empty list → no divider added (kept as passthrough)
    assert out == "PROMPT"


def test_attachment_git_diff_under_cap_renders_normally(monkeypatch):
    """Diffs under the cap render as a normal git_diff block."""
    from consult import attachments

    monkeypatch.setenv("CONSULT_ATTACHMENT_MAX_BYTES", "10000")
    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "diff --git a/x b/x\n+hello",
    )
    out = attachments.render_attachment({"source": "git_diff", "base": "main", "head": "HEAD"})
    assert "[ERROR" not in out
    assert "hello" in out
