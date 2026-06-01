"""Canonical run-slug grammar, shared across the engine.

A panellist slug is `<base>` for a single panel. Refine suffixes it per round
as `<base>.r<round_num>`, and when it re-fans a round it carries a panel index
too: `_suffix_specs` builds `<base>-<i>.r<round_num>`. These helpers are the
one place that grammar is parsed, so runner / refine / voting / strategies no
longer each re-derive it (they previously kept three copies of the round-suffix
regex plus a reverse-engineered index parser in `strategies`).
"""

from __future__ import annotations

import re

# Matches the `.r<round_num>` suffix refine appends, capturing the number.
# Capturing serves every caller: `sub` to strip it, `group(0)` for the full
# `.r<n>` text, `group(1)` for the round number.
ROUND_SUFFIX_RE = re.compile(r"\.r(\d+)$")


def strip_round(slug: str) -> str:
    """`slug` with any trailing `.r<n>` round suffix removed."""
    return ROUND_SUFFIX_RE.sub("", slug)


def round_suffix(slug: str) -> str:
    """The `.r<n>` suffix itself, or `""` when there is none.

    Used to re-attach a round marker to a freshly-derived base slug (e.g. a
    blinded `panelist-alpha`) so per-round artifacts don't collide.
    """
    m = ROUND_SUFFIX_RE.search(slug)
    return m.group(0) if m else ""


def round_number(slug: str) -> int | None:
    """The round number from a `.r<n>` suffix, or None when unsuffixed."""
    m = ROUND_SUFFIX_RE.search(slug)
    return int(m.group(1)) if m else None


def panel_index(slug: str) -> int | None:
    """The panel index `i` from a `<base>-<i>[.r<n>]` slug, or None.

    Refine's `_suffix_specs` builds round slugs as `<base>-<i>.r<round_num>`.
    Strip the round suffix, then read the trailing `-<i>` integer.
    """
    base = strip_round(slug)
    dash = base.rfind("-")
    if dash == -1:
        return None
    tail = base[dash + 1 :]
    return int(tail) if tail.isdigit() else None
