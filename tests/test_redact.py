"""Tests for `consult.redact` and the boundaries that use it.

The redaction control had zero coverage before this. Two layers here:
unit tests pin the primitive (`redact_secrets` / `redact_exc` /
`redact_traceback`), and boundary tests plant a fake key inside a provider
exception and assert it reaches none of the places a key must never go: the
returned tool result, the on-disk run artifacts, or the log.

The fake keys below are obviously not real but match the production patterns.
"""

from __future__ import annotations

import json
import logging

from consult.redact import redact_exc, redact_secrets, redact_traceback

# Match the production patterns without being real credentials.
FAKE_ANT = "sk-ant-api03-" + "A" * 40
FAKE_OPENAI = "sk-proj-" + "B" * 40
FAKE_GOOGLE = "AIza" + "C" * 35
FAKE_OPENROUTER = "sk-or-v1-" + "D" * 40


# --- redact_secrets -------------------------------------------------------


def test_redact_secrets_masks_every_key_shape():
    for key in (FAKE_ANT, FAKE_OPENAI, FAKE_GOOGLE, FAKE_OPENROUTER):
        out = redact_secrets(f"error talking to provider: {key} (401)")
        assert key not in out
        assert "[REDACTED]" in out


def test_redact_secrets_masks_authorization_header_both_cases():
    for header in ("Authorization", "authorization"):
        out = redact_secrets(f"{header}: Bearer {FAKE_OPENAI}")
        assert FAKE_OPENAI not in out


def test_redact_secrets_masks_x_api_key_header():
    out = redact_secrets(f"x-api-key: {FAKE_OPENAI}")
    assert FAKE_OPENAI not in out


def test_redact_secrets_masks_key_in_json_body():
    # litellm often echoes the upstream response body, which carries the
    # header as a lowercase JSON key. The bare-token pattern catches the key
    # itself even though the JSON quoting defeats the header-prefix pattern.
    body = json.dumps({"headers": {"authorization": f"Bearer {FAKE_ANT}"}})
    out = redact_secrets(body)
    assert FAKE_ANT not in out


def test_redact_secrets_leaves_ordinary_text_alone():
    text = "panellist claude-opus-4-7 returned status OK in 1200ms"
    assert redact_secrets(text) == text


# --- redact_exc -----------------------------------------------------------


def test_redact_exc_formats_type_and_message():
    assert redact_exc(ValueError("boom")) == "ValueError: boom"


def test_redact_exc_masks_secret_in_message():
    out = redact_exc(RuntimeError(f"auth failed: {FAKE_ANT}"))
    assert out.startswith("RuntimeError: ")
    assert FAKE_ANT not in out
    assert "[REDACTED]" in out


def test_redact_exc_redacts_before_truncating():
    # The invariant that matters: redact the full string, THEN cut. If you cut
    # first, a key straddling the cut point survives as a sub-`{20,}` fragment
    # the pattern can't match. Place the key so a 300-char cut lands 10 chars in.
    secret = "sk-ant-" + "Z" * 40
    prefix = "P" * 276  # len("RuntimeError: ")=14, +276, +10 of secret = 300
    out = redact_exc(RuntimeError(prefix + secret), limit=300)
    assert "sk-ant" not in out

    # Document why the order is load-bearing: truncate-first WOULD leak.
    naive = redact_secrets(f"RuntimeError: {prefix + secret}"[:300])
    assert "sk-ant-ZZZ" in naive


def test_redact_exc_limit_none_does_not_truncate():
    out = redact_exc(ValueError("q" * 5000), limit=None)
    assert len(out) > 5000
    assert "…" not in out


def test_redact_exc_non_positive_limit_does_not_truncate():
    # No caller passes these, but a non-positive limit must not produce the
    # nonsense `text[:-1]` slice — leave the (already redacted) string whole.
    msg = ValueError("z" * 100)
    assert redact_exc(msg, limit=0) == "ValueError: " + "z" * 100
    assert redact_exc(msg, limit=-5) == "ValueError: " + "z" * 100


# --- redact_traceback -----------------------------------------------------


def test_redact_traceback_keeps_frames_and_masks_secret():
    try:
        raise RuntimeError(f"provider 401 key={FAKE_ANT}")
    except RuntimeError as e:
        tb = redact_traceback(e)
    assert "Traceback (most recent call last)" in tb
    assert "RuntimeError" in tb
    assert FAKE_ANT not in tb
    assert "[REDACTED]" in tb


def test_redact_traceback_masks_chained_cause():
    try:
        try:
            raise ValueError(f"inner {FAKE_OPENAI}")
        except ValueError as inner:
            raise RuntimeError("outer wrapper") from inner
    except RuntimeError as e:
        tb = redact_traceback(e)
    assert FAKE_OPENAI not in tb
    assert "RuntimeError: outer wrapper" in tb


# --- boundary: synth result + on-disk synthesis.md ------------------------


async def test_synth_failure_text_redacts_secret(tmp_path, monkeypatch):
    from consult import artifacts
    from consult import synth as synth_mod
    from consult.types import ManifestEntry, RunHandle, Status

    monkeypatch.setattr(artifacts, "runs_root", lambda: tmp_path)
    paths = artifacts.create_run()
    entry = ManifestEntry(
        slug="alpha",
        model_id="anthropic/x",
        status=Status.OK,
        resource_uri=paths.resource_uri("alpha"),
        body_path=str(paths.response_text("alpha")),
        latency_ms=10,
        cost_usd=0.005,
        cost_known=True,
    )
    paths.response_text("alpha").write_text("Some response.")
    handle = RunHandle(
        run_id=paths.run_id,
        artifacts_dir=str(paths.root),
        manifest=[entry],
        cost_usd=0.005,
        cost_known=True,
        wall_ms=10,
    )
    artifacts.write_manifest(paths, handle.model_dump())

    secret = "sk-ant-" + "S" * 40

    async def boom(**kwargs):
        raise RuntimeError(f"500 from upstream: Authorization: Bearer {secret}")

    monkeypatch.setattr(synth_mod.litellm, "acompletion", boom)

    result = await synth_mod.synthesise(paths.run_id)

    assert result.status is synth_mod.SynthStatus.FAILED
    assert secret not in result.text
    assert "[REDACTED]" in result.text
    on_disk = (paths.root / "synthesis.md").read_text()
    assert secret not in on_disk


# --- boundary: refine arbiter verdict (persisted + returned) --------------


async def test_arbiter_error_redacts_secret(monkeypatch):
    from consult import refine

    secret = "sk-ant-" + "R" * 40

    async def boom(**kwargs):
        raise RuntimeError(f"401 Authorization: Bearer {secret}")

    monkeypatch.setattr(refine.litellm, "acompletion", boom)

    verdict = await refine._ask_arbiter("question", 1, [], "claude-haiku")

    assert verdict.parsed_ok is False
    assert verdict.error is not None
    assert secret not in verdict.error
    assert "[REDACTED]" in verdict.error


# --- boundary: MCP error envelope + log -----------------------------------


async def test_server_envelope_and_log_redact_secret(monkeypatch, caplog):
    from consult.mcp import server

    secret = "sk-ant-" + "Q" * 40

    async def boom(arguments, on_progress=None):
        raise RuntimeError(f"provider 401: Authorization: Bearer {secret}")

    monkeypatch.setitem(server._HANDLERS, "consult", boom)

    with caplog.at_level(logging.ERROR):
        env = await server._run_handler_with_envelopes("consult", {}, None)

    assert secret not in json.dumps(env)
    assert "[REDACTED]" in env["error"]["message"]
    assert secret not in caplog.text


def test_secret_filter_redacts_message_and_args():
    """(issue #40) The logging filter must scrub the RENDERED message, args
    included — a key carried in an arg slips past msg-only redaction."""
    import io
    import logging

    from consult.redact import SecretRedactingFilter

    key = "sk-" + "a" * 24
    logger = logging.getLogger("redact-test-filter")
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.addFilter(SecretRedactingFilter())
    try:
        logger.warning("auth failed for key %s", key)
        handler.flush()
        out = stream.getvalue()
        assert key not in out
        assert "[REDACTED]" in out
    finally:
        logger.removeHandler(handler)


def test_install_redaction_filter_is_idempotent_and_covers_handlers():
    """(issue #40) Repeat installs must not stack filters, and handlers get
    one too — handler filters are what catch records propagating up from
    child loggers."""
    import logging

    from consult.redact import SecretRedactingFilter, install_redaction_filter

    name = "redact-test-install"
    lg = logging.getLogger(name)
    h = logging.NullHandler()
    lg.addHandler(h)
    try:
        install_redaction_filter(name)
        install_redaction_filter(name)
        assert sum(isinstance(f, SecretRedactingFilter) for f in lg.filters) == 1
        assert sum(isinstance(f, SecretRedactingFilter) for f in h.filters) == 1
    finally:
        lg.removeHandler(h)


def test_scrub_exception_attrs_cleans_object_surfaces():
    """(issue #39) The exception OBJECT keeps raw message/body/args after our
    string boundaries redact; scrubbing in place protects later consumers
    (OTel record_exception, host logging) that format the object."""
    from consult.redact import scrub_exception_attrs

    key = "sk-" + "b" * 24

    class FakeProviderError(Exception):
        pass

    e = FakeProviderError(f"boom {key}")
    e.message = f"Authorization: Bearer {key}"
    e.body = {"error": f"bad key {key}", "code": 401}
    scrub_exception_attrs(e)
    assert key not in str(e)
    assert key not in e.message
    assert key not in e.body["error"]
    assert e.body["code"] == 401  # non-string values untouched


def test_scrub_exception_attrs_tolerates_readonly_attrs():
    """(issue #39) httpx-style read-only properties must not break the scrub;
    args still get cleaned so str(exc) is safe."""
    from consult.redact import scrub_exception_attrs

    key = "sk-" + "c" * 24

    class Stubborn(Exception):
        @property
        def message(self):
            return "Bearer " + key

    e = Stubborn(key)
    scrub_exception_attrs(e)  # must not raise on the read-only property
    assert key not in str(e)


def test_configure_litellm_installs_filter_on_litellm_logger():
    """(issue #40) LITELLM_LOG=DEBUG output goes through LiteLLM's own
    logger; configure_litellm must hang the redaction filter there."""
    import logging

    from consult.redact import SecretRedactingFilter
    from consult.runner import configure_litellm

    configure_litellm()
    lg = logging.getLogger("LiteLLM")
    assert any(isinstance(f, SecretRedactingFilter) for f in lg.filters)
