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
