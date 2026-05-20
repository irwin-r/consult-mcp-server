"""Recovery-tolerant JSON extraction from LLM responses.

Models sometimes wrap JSON in ``` fences, prepend prose, or both. Both the
capsule extractor and the refine arbiter expect strict-JSON output but need
to recover gracefully when the model adds noise. This helper is the single
implementation; updates (e.g. handling json5, leading prose more
aggressively) land here once.
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_FENCE_OPEN = re.compile(r"^```(?:json)?\n")
_FENCE_CLOSE = re.compile(r"\n```$")


def extract_json(text: str) -> dict[str, Any] | None:
    """Try to parse `text` as a JSON object.

    Strips a leading/trailing ```/```json fence first. On failure, searches
    for the first `{...}` block in the text and tries again. Returns None if
    no parse succeeds OR if the parsed value isn't a dict — callers do
    `data.get(...)`, so an unintended list/number/string would raise
    AttributeError and bypass the cleaner None-handling path.
    """
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_OPEN.sub("", text)
        text = _FENCE_CLOSE.sub("", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
