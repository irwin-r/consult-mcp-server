#!/usr/bin/env python3
"""Render claude -p --output-format stream-json into a readable live trace."""

import json
import sys


def emit(s):
    print(s, flush=True)


ARG_HINT_KEYS = ("description", "command", "prompt", "file_path", "pattern", "subagent_type", "query", "url")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        e = json.loads(line)
    except Exception:
        continue
    t = e.get("type")
    if t == "assistant":
        for b in e.get("message", {}).get("content", []):
            bt = b.get("type")
            if bt == "text":
                txt = (b.get("text") or "").strip()
                if txt:
                    emit(txt)
            elif bt == "tool_use":
                name = b.get("name", "?")
                inp = b.get("input", {})
                hint = ""
                if isinstance(inp, dict):
                    for k in ARG_HINT_KEYS:
                        if inp.get(k):
                            hint = str(inp[k])[:90].replace("\n", " ")
                            break
                emit(f"  → {name}: {hint}" if hint else f"  → {name}")
    elif t == "result":
        cost = e.get("total_cost_usd")
        sub = e.get("subtype", "")
        tail = f" cost=${cost}" if cost is not None else ""
        emit(f"\n■ done ({sub}){tail}")
