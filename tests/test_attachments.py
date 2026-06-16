"""Attachment rendering, git_diff resolution, trusted roots.

Split out of the old 7,400-line test_smoke.py; bodies are unchanged.
"""

from __future__ import annotations

import pytest

from consult import artifacts


def test_sources_validate_ref_rejects_shell_metachars():
    """Refs with shell metacharacters must not flow to subprocess."""
    from consult.sources import _validate_ref

    # Valid refs
    for good in ("main", "refs/heads/feature/x", "v1.0.0", "abc123", "feat+x", "HEAD~1", "HEAD^"):
        _validate_ref(good, field="base")

    # Invalid refs — anything outside [A-Za-z0-9._/+~^-] is rejected
    for bad in ("main; rm -rf /", "main$(id)", "main`whoami`", "main|cat", "main\nfoo"):
        with pytest.raises(ValueError, match="invalid base"):
            _validate_ref(bad, field="base")


def test_sources_validate_ref_rejects_leading_dash_git_option_injection():
    """SECURITY: a ref must not start with `-`, otherwise it would be
    interpreted as a git option (`base="--no-index"` becomes
    `git diff --no-index..HEAD`)."""
    from consult.sources import _validate_ref

    for bad in ("-rf", "--no-index", "-h", "--exec=evil"):
        with pytest.raises(ValueError, match="invalid base"):
            _validate_ref(bad, field="base")


def test_sources_resolve_git_diff_uses_double_dash_separator(monkeypatch, tmp_path):
    """`git diff` is invoked with a trailing `--` so a future regex
    relaxation can't smuggle an option through. Belt and braces."""
    import subprocess

    from consult import sources

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="diff body", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(tmp_path))

    sources.resolve_git_diff("main", "HEAD", repo_path=str(tmp_path))

    assert "--" in captured["cmd"], captured["cmd"]
    # `--` should be AFTER the diff range, not before
    range_idx = next(i for i, a in enumerate(captured["cmd"]) if a == "main..HEAD")
    dashdash_idx = captured["cmd"].index("--")
    assert dashdash_idx > range_idx


def test_sources_validate_repo_path_enforces_trusted_roots(tmp_path, monkeypatch):
    """`repo_path` must resolve under CONSULT_TRUSTED_REPO_ROOTS."""
    from consult.sources import _validate_repo_path

    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(trusted))

    # Inside the trusted root → returns the resolved path
    assert _validate_repo_path(str(trusted)) == trusted.resolve()

    # Outside the trusted root → raises with a helpful message
    with pytest.raises(ValueError, match="not under any CONSULT_TRUSTED_REPO_ROOTS"):
        _validate_repo_path(str(outside))


def test_sources_validate_repo_path_defaults_to_cwd(tmp_path, monkeypatch):
    """Without CONSULT_TRUSTED_REPO_ROOTS, only the current cwd is trusted."""
    from consult.sources import _validate_repo_path

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    monkeypatch.chdir(tmp_path)

    # cwd works
    sub = tmp_path / "sub"
    sub.mkdir()
    assert _validate_repo_path(str(sub)) == sub.resolve()

    # Outside cwd fails
    elsewhere = tmp_path.parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        _validate_repo_path(str(elsewhere))


def test_sources_resolve_git_diff_against_real_repo(tmp_path, monkeypatch):
    """End-to-end: init a real git repo, commit a file, modify it, and
    confirm resolve_git_diff returns the expected diff text."""
    import subprocess

    from consult.sources import resolve_git_diff

    monkeypatch.setenv("CONSULT_TRUSTED_REPO_ROOTS", str(tmp_path))
    # Init repo
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    # First commit
    (tmp_path / "hello.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "hello.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    # Change + new commit
    (tmp_path / "hello.py").write_text("print('hello, world')\n")
    subprocess.run(["git", "commit", "-aq", "-m", "change"], cwd=tmp_path, check=True)

    diff = resolve_git_diff("HEAD~1", "HEAD", repo_path=str(tmp_path))
    assert "hello.py" in diff
    assert "-print('hi')" in diff
    assert "+print('hello, world')" in diff


def test_attachment_git_diff_size_cap(monkeypatch, tmp_path):
    """git_diff source must honour CONSULT_ATTACHMENT_MAX_BYTES.

    Previously only file paths were size-capped; a multi-GB diff would
    flow straight from `sources.resolve_git_diff` into the prompt and
    OOM the server or blow the token budget.
    """
    from consult import attachments

    # Tiny cap to make the test deterministic without generating a real
    # large diff. Resolver is stubbed to return an oversized blob.
    monkeypatch.setenv("CONSULT_ATTACHMENT_MAX_BYTES", "100")
    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "x" * 5000,
    )
    out = attachments.render_attachment({"source": "git_diff", "base": "main", "head": "HEAD"})
    assert "[ERROR: diff" in out
    assert "CONSULT_ATTACHMENT_MAX_BYTES=100" in out
    # The oversized content itself must NOT appear in the rendered output.
    assert "x" * 200 not in out


def test_attachment_git_diff_empty_renders_warning_not_blank_fence(monkeypatch):
    """An empty diff (base..head with no changes) must render a directive
    WARNING marker, not a blank ```diff fence.

    A blank fence lets reviewers rubber-stamp a verdict off the surrounding
    prose because they can't tell "nothing changed" from "the prompt forgot
    the diff" (issue #76).
    """
    from consult import attachments

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "",
    )
    out = attachments.render_attachment({"source": "git_diff", "base": "main", "head": "HEAD"})
    assert "[WARNING:" in out
    assert "main..HEAD" in out
    assert "nothing to review" in out
    # No code fence — an empty diff must not render as a blank ```diff block,
    # and the no-fence marker stays invisible to the block parser/trimmer.
    assert "```" not in out


def test_attachment_git_diff_whitespace_only_renders_warning(monkeypatch):
    """A whitespace-only diff is treated as empty (after the size cap, so the
    `.strip()` can't be turned into an OOM vector)."""
    from consult import attachments

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "   \n\n  \t\n",
    )
    out = attachments.render_attachment({"source": "git_diff", "base": "v1", "head": "v2"})
    assert "[WARNING:" in out
    assert "v1..v2" in out


def test_attachment_git_diff_empty_uses_custom_label(monkeypatch):
    """The WARNING header reuses a caller's custom label, and the body still
    names the actual refs so the empty range is identifiable."""
    from consult import attachments

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "",
    )
    out = attachments.render_attachment(
        {"source": "git_diff", "base": "main", "head": "HEAD", "label": "the change"}
    )
    assert "## the change" in out
    assert "git diff main..HEAD" in out


def test_attachment_git_diff_nonempty_renders_normal_block(monkeypatch):
    """A non-empty diff is unchanged: a normal ```diff fenced block, no
    WARNING. Guards the empty-check against false positives."""
    from consult import attachments

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "diff --git a/x b/x\n+line\n",
    )
    out = attachments.render_attachment({"source": "git_diff", "base": "main", "head": "HEAD"})
    assert "[WARNING:" not in out
    assert "```diff" in out
    assert "+line" in out


def test_attachment_git_diff_size_cap_wins_over_empty_check(monkeypatch):
    """The size cap runs BEFORE the `.strip()` empty check, so a whitespace
    blob over the cap hits [ERROR: ...], not [WARNING: ...].

    This is the ordering the fix comment asserts: bound the input before
    `.strip()` touches it, so an oversized whitespace blob can't be an OOM
    vector dressed up as an empty diff.
    """
    from consult import attachments

    monkeypatch.setenv("CONSULT_ATTACHMENT_MAX_BYTES", "100")
    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: " " * 5000,
    )
    out = attachments.render_attachment({"source": "git_diff", "base": "main", "head": "HEAD"})
    assert "[ERROR: diff" in out
    assert "[WARNING:" not in out


def test_attachment_git_diff_empty_does_not_abort_other_attachments(monkeypatch, tmp_path):
    """One empty git_diff among valid attachments must not swallow the
    others. inline_attachments renders each independently, so the WARNING
    and the valid file both reach the prompt."""
    from consult import attachments

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "",
    )
    good = tmp_path / "keep.txt"
    good.write_text("real content here\n")

    out = attachments.inline_attachments(
        "question",
        [{"source": "git_diff", "base": "main", "head": "HEAD"}, str(good)],
    )
    assert "[WARNING:" in out
    assert "real content here" in out


def test_attachment_git_diff_empty_warning_evades_block_parser(monkeypatch):
    """The unfenced WARNING must not be picked up as an attachment block,
    so the trimmer and the on-disk persister leave it alone. Proves the
    no-fence claim directly rather than by inspection."""
    from consult import attachments

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "",
    )
    prompt = attachments.inline_attachments(
        "question",
        [{"source": "git_diff", "base": "main", "head": "HEAD"}],
    )
    assert "[WARNING:" in prompt
    assert attachments.extract_inlined_blocks(prompt) == []


def test_attachment_empty_file_renders_warning_not_blank_fence(tmp_path, monkeypatch):
    """A bare-string path to an empty file renders a directive WARNING marker,
    not a blank fence. Generalizes the git_diff empty-warning to plain files
    (issue #79), so a reviewer can't rubber-stamp a verdict off prose."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "empty.py"
    empty.write_text("")

    out = attachments.render_attachment(str(empty))
    assert "[WARNING:" in out
    assert "is empty or contains only whitespace" in out
    # File wording must NOT borrow the diff-specific framing.
    assert "nothing to review" not in out
    # No fence — an empty file must not render as a blank ``` block, and the
    # no-fence marker stays invisible to the block parser/trimmer.
    assert "```" not in out


def test_attachment_empty_file_dict_renders_warning_with_label(tmp_path, monkeypatch):
    """A `{path, label}` dict pointing at an empty file warns and reuses the
    caller's label in the header (`## label: path`)."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "config.yaml"
    empty.write_text("")

    out = attachments.render_attachment({"path": str(empty), "label": "the config"})
    assert "[WARNING:" in out
    assert f"## the config: {empty}" in out
    assert "```" not in out


def test_attachment_whitespace_only_file_renders_warning(tmp_path, monkeypatch):
    """A file holding only whitespace is treated as empty (the `.strip()` runs
    after `_read_text_safely` has already size-capped the content)."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\n  \t\n")

    out = attachments.render_attachment(str(blank))
    assert "[WARNING:" in out
    assert "```" not in out


def test_attachment_empty_data_kind_renders_warning(tmp_path, monkeypatch):
    """An empty `kind="data"` attachment warns too (issue #79 covers file AND
    data attachments). Stops the model inferring rows/schema from the prose."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "rows.csv"
    empty.write_text("")

    out = attachments.render_attachment({"path": str(empty), "kind": "data"})
    assert "[WARNING:" in out
    assert "```" not in out


def test_attachment_nonempty_file_renders_normal_block(tmp_path, monkeypatch):
    """A non-empty file is unchanged: a normal fenced block, no WARNING. Guards
    the empty-check against false positives."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    real = tmp_path / "real.py"
    real.write_text("print('hi')\n")

    out = attachments.render_attachment({"path": str(real), "kind": "source"})
    assert "[WARNING:" not in out
    assert "```" in out
    assert "print('hi')" in out


def test_attachment_path_dict_kind_git_diff_empty_warns_not_blank_diff_fence(tmp_path, monkeypatch):
    """A {path} dict that a non-schema library caller tags kind="git_diff" must
    still warn on empty content, NOT fall through to a blank ```diff fence.

    The git_diff *source* branch (item["source"] == "git_diff") returns early on
    empty content, so the only way kind=="git_diff" reaches the shared tail is a
    file-branch dict carrying that kind. Dropping the old `kind != "git_diff"`
    tail guard closed this bypass (raised in the #79 plan review): the guard
    discriminated on caller-supplied metadata and reintroduced the exact footgun
    #79 removes for that one malformed shape."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "empty.patch"
    empty.write_text("")

    out = attachments.render_attachment({"path": str(empty), "kind": "git_diff"})
    assert "[WARNING:" in out
    assert "```diff" not in out
    assert "```" not in out


def test_attachment_empty_file_wording_distinct_from_empty_diff(monkeypatch, tmp_path):
    """The two empty markers stay semantically separate: the file warning never
    says "nothing to review", and the diff warning never says "is empty or
    contains only whitespace". Locks the per-branch split against drift."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "empty.py"
    empty.write_text("")
    file_out = attachments.render_attachment(str(empty))

    monkeypatch.setattr(
        attachments.sources,
        "resolve_git_diff",
        lambda base, head, repo_path: "",
    )
    diff_out = attachments.render_attachment({"source": "git_diff", "base": "main", "head": "HEAD"})

    assert "nothing to review" in diff_out
    assert "nothing to review" not in file_out
    assert "is empty or contains only whitespace" in file_out
    assert "is empty or contains only whitespace" not in diff_out


def test_attachment_empty_file_does_not_abort_other_attachments(tmp_path, monkeypatch):
    """One empty file among valid attachments must not swallow the others.
    inline_attachments renders each independently, so the WARNING and the valid
    file both reach the prompt."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "empty.py"
    empty.write_text("")
    good = tmp_path / "keep.txt"
    good.write_text("real content here\n")

    out = attachments.inline_attachments("question", [str(empty), str(good)])
    assert "[WARNING:" in out
    assert "real content here" in out


def test_attachment_empty_file_warning_evades_block_parser(tmp_path, monkeypatch):
    """The unfenced empty-file WARNING must not be picked up as an attachment
    block, so the trimmer and the on-disk persister leave it alone."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    empty = tmp_path / "empty.py"
    empty.write_text("")

    prompt = attachments.inline_attachments("question", [str(empty)])
    assert "[WARNING:" in prompt
    assert attachments.extract_inlined_blocks(prompt) == []


def test_attachment_oversized_whitespace_file_is_error_not_warning(tmp_path, monkeypatch):
    """The size cap (in `_read_text_safely`, via stat) runs before content is
    read, so an oversized whitespace file hits [ERROR: ...], not [WARNING: ...].
    Mirrors the git_diff size-cap-wins ordering for the file path."""
    from consult import attachments

    monkeypatch.delenv("CONSULT_TRUSTED_REPO_ROOTS", raising=False)
    monkeypatch.setenv("CONSULT_ATTACHMENT_MAX_BYTES", "10")
    blank = tmp_path / "big_blank.txt"
    blank.write_text(" " * 5000)

    out = attachments.render_attachment(str(blank))
    assert "[ERROR:" in out
    assert "CONSULT_ATTACHMENT_MAX_BYTES=10" in out
    assert "[WARNING:" not in out


def test_persist_inlined_attachments_writes_and_uri_resolves(tmp_path, monkeypatch):
    """Persistence writes each block's content to attachments/<safe-name>
    and the returned URI is resolvable via parse_resource_uri."""
    from consult import attachments as att

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    prompt = (
        "review this:\n"
        + att.ATTACHMENT_SEPARATOR
        + "\n# /Users/x/foo.py\n```python\nbody-foo\n```\n"
        + "\n# /Users/x/bar.py\n```python\nbody-bar\n```\n"
        + "\n# /Users/x/foo.py\n```python\nbody-foo-2\n```\n"  # name collision
    )
    uri_map = att.persist_inlined_attachments(paths, prompt)
    assert len(uri_map) == 3
    files = sorted(p.name for p in paths.attachments.iterdir())
    assert "foo.py" in files
    assert "bar.py" in files
    # Collision got a numeric suffix.
    assert any(f.startswith("foo.py-") for f in files)
    for uri in uri_map.values():
        rid, kind, name = artifacts.parse_resource_uri(uri)
        assert rid == paths.run_id
        assert kind == "attachments"
        assert (paths.attachments / name).exists()


@pytest.mark.asyncio
async def test_fit_prompt_drops_largest_attachment_with_stub(tmp_path, monkeypatch):
    """When the prompt is over budget, the trimmer drops the LARGEST
    attachment block first, replaces it with a stub referencing the
    resource URI, and leaves smaller blocks intact. Beats head+tail
    slicing through the middle of a code file."""
    from consult import attachments as att
    from consult import runner

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text),
    )

    small_block = "small\n" * 50  # ~300 chars
    large_block = "X" * 50_000  # 50K chars — dropped first
    medium_block = "Y" * 5_000  # 5K chars

    prompt = (
        "instructions here\n"
        + att.ATTACHMENT_SEPARATOR
        + f"\n# /src/small.py\n```python\n{small_block}\n```\n"
        + f"\n# /src/big.py\n```python\n{large_block}\n```\n"
        + f"\n# /src/medium.py\n```python\n{medium_block}\n```\n"
    )
    att.persist_inlined_attachments(paths, prompt)

    out, dropped = await runner._fit_prompt_to_context(
        prompt,
        paths=paths,
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=11_000,
        max_output_tokens=1_000,
    )
    assert dropped > 0
    assert large_block not in out
    assert "Attachment dropped to fit context" in out
    assert "big.py" in out  # header preserved
    assert paths.run_id in out  # resource URI for the dropped file
    assert small_block in out


@pytest.mark.asyncio
async def test_fit_prompt_falls_back_to_head_tail_without_attachments(monkeypatch):
    """Prompts that aren't structured as attachments still get the
    head+tail trim — the new path is additive, not a replacement."""
    from consult import runner

    monkeypatch.setattr(
        runner.litellm,
        "token_counter",
        lambda model, text: len(text),
    )
    long_prompt = "A" * 1000 + "B" * 1000  # no ATTACHMENT_SEPARATOR
    out, dropped = await runner._fit_prompt_to_context(
        long_prompt,
        paths=None,
        prior_turns=None,
        litellm_id="x/y",
        max_input_tokens=1000,
        max_output_tokens=100,
    )
    assert "[TRIMMED" in out
    assert dropped > 0
