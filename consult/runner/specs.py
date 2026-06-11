"""Panel-spec grammar: `model:N` expansion, slug derivation, prompt assembly."""

from __future__ import annotations

import logging
import os
import re

from .. import slugs
from ..types import ModelSpec

logger = logging.getLogger(__name__)


# CONTRACT: capsule.py:_CONFIDENCE and the capsule extractor prompt depend on
# these exact line prefixes (`CONFIDENCE:` and `KEY_REASON:`). Don't rename
# either without updating both.
_FOOTER = """\
---
End your response with EXACTLY these two lines (after your main answer):
CONFIDENCE: <number between 0.0 and 1.0 reflecting how sure you are>
KEY_REASON: <one sentence — the single most important reason for your view>"""


def _build_per_slug_prompt(base_prompt: str, stance_prompt: str) -> str:
    head = f"{stance_prompt}\n\n" if stance_prompt else ""
    return f"{head}{base_prompt}\n\n{_FOOTER}"


_MODEL_COUNT_SUFFIX = re.compile(r"^(.+):(\d+)$")

# Upper bound on the number of panellists one fanout may run. `expand_specs`
# turns caller-supplied `model:N` sugar into N specs, and N arrives from
# untrusted MCP input; without a cap a single `model:1000000000` builds a
# billion specs and OOMs the process before the cost gate can reject it. The
# largest built-in tier is `deep` at 14, so 64 leaves generous headroom for
# stochastic averaging. Override with CONSULT_MAX_PANEL_SIZE.
_DEFAULT_MAX_PANEL_SIZE = 64

# Longest `model:N` count string we'll hand to int(). A handful of digits
# covers any sane cap; the guard rejects a pathological all-digits count
# before it reaches the quadratic str->int path (CVE-2020-10735) — relevant
# on pre-3.11 runtimes and non-CPython, which lack the interpreter-level
# digit limit.
_MAX_COUNT_DIGITS = 9


def _max_panel_size() -> int:
    """Resolve the panel-size cap, honouring CONSULT_MAX_PANEL_SIZE.

    Read per call (not frozen at import) so tests and operators can change it
    via the environment, matching the `CONSULT_ATTACHMENT_MAX_BYTES` /
    `CONSULT_MAX_CONCURRENCY` idiom. An invalid or non-positive override falls
    back to the default with a warning rather than silently disabling the
    guard.
    """
    raw = os.environ.get("CONSULT_MAX_PANEL_SIZE", "").strip()
    if not raw:
        return _DEFAULT_MAX_PANEL_SIZE
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "CONSULT_MAX_PANEL_SIZE=%r is not an int; using %d",
            raw,
            _DEFAULT_MAX_PANEL_SIZE,
        )
        return _DEFAULT_MAX_PANEL_SIZE
    if val < 1:
        logger.warning(
            "CONSULT_MAX_PANEL_SIZE=%r must be ≥1; using %d",
            raw,
            _DEFAULT_MAX_PANEL_SIZE,
        )
        return _DEFAULT_MAX_PANEL_SIZE
    return val


# Raw LiteLLM IDs can legitimately contain characters the slug regex
# rejects — e.g. OpenRouter's `:free` suffix. Sanitise model-derived slugs
# so a valid model alias never crashes the path-build downstream. The
# user-supplied slug path is unchanged: that goes through ModelSpec's
# field_validator which fails fast at the input boundary.
_SLUG_BAD_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitise_derived_slug(base: str) -> str:
    """Coerce a model-derived slug fragment into the safe-id character set.

    Multiple bad characters in a row collapse to a single `-` and any
    leading/trailing `-`/`.` are stripped so the result is also a legal
    leading character (the slug regex anchors on an alphanumeric).
    """
    out = _SLUG_BAD_CHARS_RE.sub("-", base)
    return out.strip("-.")


def expand_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Expand `model:N` syntax into N copies of the spec.

    A trailing `:<positive int>` on `spec.model` requests N instances of the
    same model in the panel — useful for stochastic averaging (run the same
    prompt N times and compare), or to grow a panel without adding new
    aliases. The slug-disambiguation in `_make_slug` already appends an
    index suffix when the same base name repeats, so no extra work needed
    downstream.

    Idempotent: passing already-expanded specs (no `:N` suffix on any
    `model`) returns them unchanged. Bare model strings with internal
    colons that aren't followed by digits (rare, but possible for raw
    LiteLLM IDs) are left alone — the regex anchors to the end.

    Raises `ValueError` if N < 1 (zero copies is almost certainly a typo),
    if a single `model:N` exceeds the panel-size cap, or if the specs
    together exceed it. N is caller-supplied, so the cap is enforced before
    the oversized list is built — an unbounded `range(N)` would OOM the
    process ahead of the cost gate. The cap is `CONSULT_MAX_PANEL_SIZE`
    (default 64); see `_max_panel_size`.
    """
    cap = _max_panel_size()
    out: list[ModelSpec] = []
    for spec in specs:
        m = _MODEL_COUNT_SUFFIX.match(spec.model)
        if not m:
            out.append(spec)
        else:
            base, count_str = m.group(1), m.group(2)
            # Strip leading zeros so the length check judges magnitude, not
            # padding: a legal but zero-padded ":0000064" must not read as
            # "implausibly large". After the strip an empty string is all
            # zeros (count 0), and a too-long run is rejected before int()
            # reaches the quadratic str->int path (CVE-2020-10735).
            digits = count_str.lstrip("0")
            if not digits:
                raise ValueError(f"model:count must be ≥1 (got {spec.model!r})")
            if len(digits) > _MAX_COUNT_DIGITS:
                raise ValueError(f"model:count is implausibly large (got {spec.model!r})")
            count = int(digits)  # ≥1: leading zeros stripped, non-empty
            if count > cap:
                raise ValueError(
                    f"model:count {count} exceeds the {cap}-panellist cap "
                    f"(CONSULT_MAX_PANEL_SIZE); got {spec.model!r}"
                )
            # When an explicit slug is set and count > 1, suffix each copy
            # with its index. Without this all N copies share the same slug,
            # race to write to the same `responses/<slug>.txt`, and N-1
            # responses are silently overwritten. The bare-model branch is
            # safe because `_make_slug` already disambiguates by index.
            for i in range(count):
                slug = f"{spec.slug}-{i}" if (spec.slug and count > 1) else spec.slug
                out.append(ModelSpec(model=base, stance=spec.stance, slug=slug))
        # Check after every spec, not just at the end: many specs each under
        # the per-spec cap can still sum past it, and we want to fail before
        # the accumulated list grows large (each spec adds at most `cap`, so
        # the transient list never exceeds 2*cap).
        if len(out) > cap:
            raise ValueError(f"panel has more than {cap} panellists (CONSULT_MAX_PANEL_SIZE)")
    return out


def _make_slug(spec: ModelSpec, idx: int, blinded: bool) -> str:
    """Single-spec slug derivation. Prefer `_make_slugs` for whole panels;
    this exists for backwards-compat and is used by tests that construct
    one entry at a time.
    """
    if blinded:
        # alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu
        greek = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
        ]
        base = f"panelist-{greek[idx]}" if idx < len(greek) else f"panelist-{idx}"
        # Preserve refine's `.r<n>` round suffix even when blinded so the
        # round-N artifact doesn't overwrite round-(N-1)'s. Without this,
        # blinded refine writes every round to the same `panelist-alpha.txt`
        # file and the per-round transcript is destroyed.
        if spec.slug:
            # Preserve the `.r<n>` round suffix (round_suffix is "" if absent).
            base = f"{base}{slugs.round_suffix(spec.slug)}"
        return base
    if spec.slug:
        return spec.slug
    # Derive a stable, readable slug from the alias/id. Raw LiteLLM IDs may
    # contain characters the safe-id regex rejects (e.g. `:free` on
    # OpenRouter); sanitise so a valid model never crashes the path build.
    base = sanitise_derived_slug(spec.model.split("/")[-1].lower())
    return f"{base}-{idx}" if idx > 0 else base


def _make_slugs(specs: list[ModelSpec], blinded: bool) -> list[str]:
    """Compute slugs for a whole panel with per-base disambiguation.

    Single-instance models get the bare base name. Only repeats of the
    same base get a `-N` index suffix, so `[claude-opus, gpt-pro]` yields
    `["claude-opus", "gpt-pro"]` instead of the previous global-index
    `["claude-opus", "gpt-pro-1"]` (unearned suffix on the only gpt-pro).

    The refine `.r<n>` round suffix and the blinded greek-letter slug
    flow still go through `_make_slug` per spec — only the non-blinded
    bare-model path needs panel-level awareness.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, spec in enumerate(specs):
        if blinded or spec.slug:
            out.append(_make_slug(spec, i, blinded))
            continue
        base = sanitise_derived_slug(spec.model.split("/")[-1].lower())
        count = seen.get(base, 0)
        out.append(f"{base}-{count}" if count else base)
        seen[base] = count + 1
    return out
