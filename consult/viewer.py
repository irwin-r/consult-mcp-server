"""Render a complete consult run as a self-contained HTML page.

Reads every artifact under `~/.consult/runs/<run_id>/` (or
`$CONSULT_RUNS_DIR`) and writes a single `feed.html` next to them. No
external assets, no JavaScript, no live tail — the file is fully usable
offline.

Console entry point:
  consult-view <run_id>          # generate feed.html, print its path
  consult-view <run_id> --open   # also open it in the default browser
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from . import artifacts

logger = logging.getLogger(__name__)


# ---- Minimal markdown -------------------------------------------------------
# `synthesis.md` is the only markdown source the viewer renders. A full library
# would be a new dep for a thin slice of features the synthesiser actually
# emits (headings, bullet/numbered lists, **bold**, *italic*, `code`, links,
# fenced code, horizontal rules, paragraphs). Doing it in ~70 lines also lets
# us guarantee every byte of untrusted source is HTML-escaped before any
# inline transform runs — a panellist body that contained `<script>` would
# otherwise become live HTML.

_MD_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_MD_BULLET = re.compile(r"^\s*[-*]\s+(.+)$")
_MD_NUMBER = re.compile(r"^\s*\d+\.\s+(.+)$")
_MD_HR = re.compile(r"^-{3,}\s*$")


def _md_inline(escaped: str) -> str:
    # Stash inline-code spans before bold/italic so a literal `**` inside a
    # code span isn't reinterpreted as bold markup.
    stash: list[str] = []

    def _save(m: re.Match[str]) -> str:
        stash.append(m.group(1))
        return f"\x00C{len(stash) - 1}\x00"

    out = _MD_INLINE_CODE.sub(_save, escaped)
    out = _MD_BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _MD_ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = _MD_LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        out,
    )
    return re.sub(
        r"\x00C(\d+)\x00",
        lambda m: f"<code>{stash[int(m.group(1))]}</code>",
        out,
    )


def render_markdown(src: str) -> str:
    """Markdown subset → HTML. Untrusted source is safe: each line is
    HTML-escaped before inline transforms run.
    """
    if not src or not src.strip():
        return ""
    out: list[str] = []
    list_stack: list[str] = []
    para: list[str] = []
    code_buf: list[str] = []
    in_code = False
    code_lang = ""

    def _flush_para() -> None:
        if para:
            out.append(f"<p>{_md_inline(html.escape(' '.join(para)))}</p>")
            para.clear()

    def _close_lists() -> None:
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    for raw in src.splitlines():
        if raw.startswith("```"):
            if in_code:
                cls = f' class="lang-{html.escape(code_lang)}"' if code_lang else ""
                body = html.escape("\n".join(code_buf))
                out.append(f"<pre><code{cls}>{body}</code></pre>")
                code_buf.clear()
                code_lang = ""
                in_code = False
            else:
                _flush_para()
                _close_lists()
                code_lang = raw[3:].strip()
                in_code = True
            continue
        if in_code:
            code_buf.append(raw)
            continue
        if not raw.strip():
            _flush_para()
            _close_lists()
            continue
        if _MD_HR.match(raw):
            _flush_para()
            _close_lists()
            out.append("<hr>")
            continue
        m = _MD_HEADING.match(raw)
        if m:
            _flush_para()
            _close_lists()
            level = len(m.group(1))
            out.append(f"<h{level}>{_md_inline(html.escape(m.group(2)))}</h{level}>")
            continue
        m = _MD_BULLET.match(raw)
        if m:
            _flush_para()
            if not list_stack or list_stack[-1] != "ul":
                _close_lists()
                list_stack.append("ul")
                out.append("<ul>")
            out.append(f"<li>{_md_inline(html.escape(m.group(1)))}</li>")
            continue
        m = _MD_NUMBER.match(raw)
        if m:
            _flush_para()
            if not list_stack or list_stack[-1] != "ol":
                _close_lists()
                list_stack.append("ol")
                out.append("<ol>")
            out.append(f"<li>{_md_inline(html.escape(m.group(1)))}</li>")
            continue
        para.append(raw.strip())

    _flush_para()
    _close_lists()
    if in_code:  # unterminated fence — keep contents readable
        out.append(f"<pre><code>{html.escape(chr(10).join(code_buf))}</code></pre>")
    return "\n".join(out)


# ---- Helpers ----------------------------------------------------------------

_STATUS_TONE = {
    "OK": "ok",
    "TRUNCATED": "warn",
    "MALFORMED": "err",
    "EMPTY": "muted",
    "REFUSED": "err",
    "CONTENT_FILTERED": "err",
    "RATE_LIMITED": "warn",
    "TIMEOUT": "warn",
    "ERROR": "err",
    "SKIPPED": "muted",
}


def _status_tone(status: str) -> str:
    return _STATUS_TONE.get(status, "muted")


def _fmt_cost(usd: float | None, known: bool = True) -> str:
    if usd is None:
        return "—" if known else "?"
    if usd == 0:
        s = "$0"
    elif usd < 1.0:
        s = f"${usd:.4f}".rstrip("0").rstrip(".")
    else:
        s = f"${usd:.2f}"
    return s if known else f"{s}*"


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


_ROUND_SUFFIX = re.compile(r"\.r(\d+)$")


def _round_of(slug: str) -> int | None:
    m = _ROUND_SUFFIX.search(slug)
    return int(m.group(1)) if m else None


def _pill(text: str, tone: str = "muted") -> str:
    return f'<span class="pill pill-{tone}">{html.escape(text)}</span>'


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # `Z` suffix isn't valid ISO before 3.11 fully; the JSONL writer
        # already emits `+00:00`, but accept both for forward compat.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# ---- Provider branding ------------------------------------------------------
# Each panellist's model_id is rendered as a colour-coded brand badge + a
# humanised display name. The badge maps litellm provider segments (and the
# openrouter `provider/sub-provider/...` routing) to a CSS class with brand
# colour, plus a short letter glyph. Unknown providers fall through to a
# generic "other" badge so the page still renders for newly-added models.

_PROVIDER_KEY = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "gemini": "google",
    "vertex_ai": "google",
    "x-ai": "xai",
    "xai": "xai",
    "deepseek": "deepseek",
    "mistralai": "mistral",
    "mistral": "mistral",
    "meta-llama": "meta",
    "meta": "meta",
    "perplexity": "perplexity",
    "qwen": "qwen",
    "alibaba": "qwen",
    "moonshotai": "moonshot",
    "moonshot": "moonshot",
    "zhipuai": "zhipu",
    "z-ai": "zhipu",
    "zai": "zhipu",
    "xiaomi": "xiaomi",
}

# (badge_glyph, display_name) per provider key — the badge is small so we
# keep glyphs to 1–2 chars. Display name surfaces in the card head next to
# the badge.
_PROVIDER_INFO = {
    "anthropic": ("A", "Anthropic"),
    "openai": ("O", "OpenAI"),
    "google": ("G", "Google"),
    "xai": ("X", "xAI"),
    "deepseek": ("D", "DeepSeek"),
    "mistral": ("M", "Mistral"),
    "meta": ("L", "Meta"),
    "perplexity": ("P", "Perplexity"),
    "qwen": ("Q", "Qwen"),
    "moonshot": ("K", "Moonshot"),
    "zhipu": ("Z", "Zhipu"),
    "xiaomi": ("Mi", "Xiaomi"),
    "other": ("?", ""),
}


# Provider logo paths — Simple Icons (CC0) for the brands we can fairly
# represent at a 12×12 silhouette. Each value is just the path data; the
# `<symbol>` wrapper is added in `_brand_sprite_html`. Providers absent
# from this map fall back to a centred letter glyph via `_PROVIDER_INFO`
# so every badge still renders through the same SVG mechanism.
_PROVIDER_SVG_PATHS = {
    "anthropic": (
        "M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442"
        "l1.3693-3.5527h7.0052l1.3693 3.5527h3.7442L10.5363 3.541Zm-.3712 "
        "10.2232L8.6087 7.8208l2.2914 5.9434Z"
    ),
    "openai": (
        "M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0"
        " 0-6.5098-2.9001A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 "
        "0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 "
        ".511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 "
        "13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 "
        "3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a"
        "4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948 "
        ".7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038."
        "052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zM3.6 18.3038a4.4708 "
        "4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 "
        ".7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 "
        "19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 "
        "4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 "
        "3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 "
        "4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 "
        "7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 "
        "8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852"
        "-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662."
        "0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 "
        "4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742"
        "a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948."
        "7948 0 0 0-.3927.6813zm1.0976-2.3654 2.602-1.4998 2.6069 1.4998v"
        "2.9994l-2.5974 1.4997-2.6067-1.4997Z"
    ),
    "google": (
        "M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 "
        "1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 "
        "8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 "
        "16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 "
        "0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053"
        "-1.467-.173-2.053H12.48z"
    ),
    "xai": (
        "M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 "
        "7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932Z"
    ),
    "meta": (
        "M6.915 4.03c-1.968 0-3.683 1.28-4.871 3.113C.704 9.208 0 11.883 0 "
        "14.449c0 .706.07 1.369.21 1.973a6.624 6.624 0 0 0 .265.86 5.297 "
        "5.297 0 0 0 .371.761c.696 1.159 1.818 1.927 3.593 1.927 1.497 0 "
        "2.633-.671 3.965-2.444.76-1.012 1.144-1.626 2.663-4.32l.756-1.339."
        "186-.325c.061.1.121.196.183.3l2.152 3.595c.724 1.21 1.665 2.556 "
        "2.47 3.314 1.046.987 1.992 1.22 3.06 1.22 1.075 0 1.876-.355 "
        "2.455-.843a3.743 3.743 0 0 0 .81-.973c.542-.939.861-2.127.861"
        "-3.745 0-2.72-.681-5.357-2.084-7.45-1.282-1.912-2.957-2.93-4.716"
        "-2.93-1.047 0-2.088.467-3.053 1.308-.652.57-1.257 1.29-1.82 2.05"
        "-.69-.875-1.335-1.547-1.958-2.056-1.182-.966-2.315-1.303-3.454"
        "-1.303zm10.16 2.053c1.147 0 2.188.758 2.992 1.999 1.132 1.748 "
        "1.647 4.195 1.647 6.4 0 1.548-.368 2.9-1.839 2.9-.58 0-1.027-.23"
        "-1.664-1.004-.496-.601-1.343-1.878-2.832-4.358l-.617-1.028a44.908 "
        "44.908 0 0 0-1.255-1.98c.07-.109.141-.224.211-.327 1.12-1.667 "
        "2.118-2.602 3.358-2.602zm-10.201.553c1.265 0 2.058.791 2.675 "
        "1.446.307.327.737.871 1.234 1.579l-1.02 1.566c-.757 1.163-1.882 "
        "3.017-2.837 4.338-1.191 1.649-1.81 1.817-2.486 1.817-.524 0-1.038"
        "-.237-1.383-.794-.263-.426-.464-1.13-.464-2.046 0-2.221.63-4.535 "
        "1.66-6.088.454-.687.964-1.226 1.533-1.533a2.264 2.264 0 0 1 "
        "1.088-.285z"
    ),
    "perplexity": (
        "M22.3977 7.0896h-2.3106V.0676l-7.3046 6.3542V.1577h-1.3433v6.1966"
        "L4.4904.0676v7.022H2.1797v10.0093h2.3107v6.8334l7.3091-6.0816v6."
        "0816h1.3433v-6.0816l7.3046 6.0816V17.121h2.3503zM4.4904 1.8125 "
        "11.3148 7.795 4.4904 13.7775zm-1.0535 6.6963h8.9009v8.0042H3.4369z"
        "m9.9544 9.6916V18.43l5.9603 4.9588zm0-3.3252V8.0007l6.8244 5.9774z"
        "m6.8205-13.4627v6.0987L13.391 7.0896z"
    ),
    "xiaomi": ("M22 22V2H2v20h4.61V6.65h10.78V22H22zm-9.42 0V10.55H7.97V22h4.61z"),
}


# Bare-alias prefixes (no provider segment) → provider key. Used when the
# caller passes a registry alias like `gemini-pro` instead of a litellm id.
_ALIAS_PREFIXES = [
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("deepseek", "deepseek"),
    ("mistral", "mistral"),
    ("llama", "meta"),
    ("sonar", "perplexity"),
    ("perplexity", "perplexity"),
    ("qwen", "qwen"),
    ("kimi", "moonshot"),
    ("moonshot", "moonshot"),
    ("glm", "zhipu"),
    ("mimo", "xiaomi"),
]


def _provider_of(model_id: str | None) -> tuple[str, str]:
    """Return (provider_key, model_part). `model_part` is everything after
    the provider prefix — what we humanise for display.

    Handles four shapes:
      - `provider/model`        (anthropic/claude-opus-4-7)
      - `openrouter/sub/model`  (openrouter/x-ai/grok-4.3)
      - bare alias              (gemini-pro, claude-opus — used for the
                                 synth-model badge in the header)
      - unknown                 (falls through to the generic badge)
    """
    if not model_id:
        return ("other", "")
    if "/" in model_id:
        parts = model_id.split("/")
        if parts[0] == "openrouter" and len(parts) >= 3:
            sub = parts[1].lower()
            model_part = "/".join(parts[2:])
        else:
            sub = parts[0].lower()
            model_part = "/".join(parts[1:])
        return (_PROVIDER_KEY.get(sub, "other"), model_part)
    low = model_id.lower()
    for prefix, key in _ALIAS_PREFIXES:
        if low.startswith(prefix):
            return (key, model_id)
    return ("other", model_id)


# Brand display overrides for terms that lose their casing when split on `-`.
# Anything not here gets simple title-casing in `_humanise_model`.
_BRAND_TERMS = {
    "gpt": "GPT",
    "glm": "GLM",
    "mimo": "MiMo",
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "xai": "xAI",
}


def _humanise_model(model_part: str) -> str:
    """Heuristic prettier display of a model name like
    `claude-opus-4-7` → `Claude Opus 4.7`, `gpt-5.5-pro` → `GPT 5.5 Pro`.

    Rules:
      - dash between two digits becomes a dot (4-7 → 4.7)
      - trailing 6+ digit run is treated as a date stamp and dropped
      - `-preview` / `-latest` suffixes dropped
      - dash-separated tokens: brand overrides first, then numeric kept
        as-is, otherwise capitalise
    """
    if not model_part:
        return ""
    # Strip the date stamp first — once dashes become dots below, the
    # trailing run wouldn't be reachable as `-NNNNNN` any more.
    s = re.sub(r"-\d{6,}$", "", model_part)
    s = re.sub(r"(?<=\d)-(?=\d)", ".", s)
    for suffix in ("-preview", "-latest", "-exp"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    out = []
    for raw in s.split("-"):
        low = raw.lower()
        if low in _BRAND_TERMS:
            out.append(_BRAND_TERMS[low])
            continue
        # Tokens shaped like a version (optional single-letter prefix + digits):
        # `v4` stays lowercase (industry convention for version prefixes), but
        # things like `k2.6` (Kimi's model family marker, not a version) get
        # title-cased so the brand reads correctly.
        if re.fullmatch(r"[a-z]?\d+(\.\d+)*", low):
            if low.startswith("v"):
                out.append(low)
            elif low[:1].isalpha():
                out.append(low[0].upper() + low[1:])
            else:
                out.append(low)
            continue
        out.append(raw[:1].upper() + raw[1:] if raw else raw)
    return " ".join(out)


def _brand_symbol_body(provider: str) -> str:
    """Inner content for the provider's `<symbol>` in the page's SVG
    sprite — either a real logo path or a centred letter glyph for the
    providers we don't have logo art for. Both render through the same
    `<use href>` mechanism so all badges look consistent.
    """
    if provider in _PROVIDER_SVG_PATHS:
        return f'<path d="{_PROVIDER_SVG_PATHS[provider]}"/>'
    glyph, _ = _PROVIDER_INFO.get(provider, ("?", ""))
    # Two-character glyphs (e.g. Xiaomi's "Mi" — which is already in
    # `_PROVIDER_SVG_PATHS`, but defensive) get a smaller font.
    size = 12 if len(glyph) > 1 else 15
    return (
        f'<text x="12" y="17" text-anchor="middle" '
        f'font-family="-apple-system,BlinkMacSystemFont,Roboto,sans-serif" '
        f'font-size="{size}" font-weight="700" fill="currentColor">'
        f"{html.escape(glyph)}</text>"
    )


def _brand_sprite_html() -> str:
    """Emit a single hidden SVG once per page; each `.brand` badge then
    references the relevant symbol by id, so the logo path data isn't
    duplicated per citation pill (a refine run page can contain dozens).
    """
    providers = sorted(set(_PROVIDER_INFO) | set(_PROVIDER_SVG_PATHS))
    symbols = "".join(
        f'<symbol id="brand-{p}" viewBox="0 0 24 24">{_brand_symbol_body(p)}</symbol>' for p in providers
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'style="position:absolute;width:0;height:0;overflow:hidden" '
        f'aria-hidden="true">{symbols}</svg>'
    )


def _brand_glyph_html(provider: str) -> str:
    """A `<use>` reference to the provider's symbol in the page sprite."""
    return (
        f'<svg class="brand-icon" viewBox="0 0 24 24" aria-hidden="true">'
        f'<use href="#brand-{provider}"/></svg>'
    )


_GREEK_LETTERS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
}

# Slug shape produced by runner._make_slug under blinded=True. The optional
# `.r<n>` suffix preserves the round identity in refine runs.
_BLINDED_SLUG_RE = re.compile(r"^panelist-([a-z]+)(?:\.r\d+)?$")


def _blinded_greek_for_slug(slug: str | None) -> str | None:
    """Return the greek glyph for a blinded panelist slug, or None for
    non-blinded slugs. `panelist-alpha.r1` → `α`.
    """
    if not slug:
        return None
    m = _BLINDED_SLUG_RE.match(slug)
    return _GREEK_LETTERS.get(m.group(1)) if m else None


def _blinded_inline_glyph(letter: str) -> str:
    """Inline SVG text glyph for a blinded panelist — bypasses the brand
    sprite so each blinded panelist gets a distinct letter without
    pre-baking 12 symbols. Matches `_brand_symbol_body`'s text styling.
    """
    return (
        f'<svg class="brand-icon" viewBox="0 0 24 24" aria-hidden="true">'
        f'<text x="12" y="17" text-anchor="middle" '
        f'font-family="-apple-system,BlinkMacSystemFont,Roboto,sans-serif" '
        f'font-size="15" font-weight="700" fill="currentColor">'
        f"{html.escape(letter)}</text></svg>"
    )


def _model_badge_html(model_id: str | None, slug: str | None = None) -> str:
    """Inline badge + humanised name. Falls back to the raw id when the
    provider isn't in the map so nothing disappears silently. For blinded
    panels (model_id=None, slug like `panelist-alpha`), uses the greek
    letter as the glyph so each panelist stays visually distinct.
    """
    greek = _blinded_greek_for_slug(slug) if model_id is None else None
    if greek is not None:
        title = html.escape(f"blinded panellist · {slug}")
        return (
            f'<span class="model-tag" title="{title}">'
            f'<span class="brand brand-other">{_blinded_inline_glyph(greek)}</span>'
            f'<span class="model-name">Panellist {html.escape(greek)}</span>'
            f"</span>"
        )
    provider, model_part = _provider_of(model_id)
    _, brand_name = _PROVIDER_INFO[provider]
    display = _humanise_model(model_part) or (model_id or "—")
    title = html.escape(model_id or "unknown model")
    brand_attr = f"{brand_name} · " if brand_name else ""
    return (
        f'<span class="model-tag" title="{brand_attr}{title}">'
        f'<span class="brand brand-{provider}">{_brand_glyph_html(provider)}</span>'
        f'<span class="model-name">{html.escape(display)}</span>'
        f"</span>"
    )


# Bracketed citation pattern: `[slug]` or `[slug1, slug2, ...]`. Operates on
# already-HTML-escaped text, since the synthesis is rendered to HTML first
# and any non-citation brackets at that point are literal characters
# (markdown links `[text](url)` were already consumed by `render_markdown`).
_CITATION_PATTERN = re.compile(r"\[([\w.\- ]+(?:,\s*[\w.\- ]+)*)\]")


def _cite_pill(
    slug: str,
    model_id: str | None,
    anchor_slug: str | None = None,
) -> str:
    """Render a single slug as a coloured citation pill — brand badge +
    humanised model name + optional `rN` round chip. Shared between
    `_decorate_citations` (for synthesis/arbiter prose) and the capsule
    block's agrees_with/disagrees_with rows.

    `anchor_slug` overrides the href/round-chip when the cited slug is a
    bare alias (e.g. `claude-haiku-2`) and we want the pill to jump to the
    canonical round-suffixed card (`claude-haiku-2.r3`).
    """
    if anchor_slug is None:
        anchor_slug = slug
    # Round chip reflects the resolved target, not the bare citation text,
    # so `[claude-haiku-2]` → `claude-haiku-2.r3` still shows `r3`.
    round_num = _round_of(anchor_slug)
    round_chip = f' <span class="cite-round">r{round_num}</span>' if round_num is not None else ""
    # Anchor to the matching panellist card so citations are click-to-jump.
    # Slugs are `[a-zA-Z0-9._-]+` by registry convention, so html.escape is
    # sufficient — no URL-encoding edge cases.
    href = "#card-" + html.escape(anchor_slug, quote=True)
    greek = _blinded_greek_for_slug(anchor_slug) if model_id is None else None
    if greek is not None:
        title = html.escape(f"blinded panellist · {slug}")
        return (
            f'<a href="{href}" class="cite cite-other" title="{title}">'
            f'<span class="brand brand-other">{_blinded_inline_glyph(greek)}</span>'
            f'<span class="cite-name">Panellist {html.escape(greek)}</span>'
            f"{round_chip}"
            f"</a>"
        )
    provider, model_part = _provider_of(model_id)
    _, brand_name = _PROVIDER_INFO[provider]
    # Prefer the humanised model name over the raw slug — "Claude Opus 4.7"
    # reads better than "opus-0.r1". Falls back to the slug when there's no
    # model_id to humanise (e.g. a citation that doesn't match any
    # panellist in the manifest).
    display = _humanise_model(model_part) if model_id else slug
    if not display:
        display = slug
    # Tooltip carries the raw slug + full model_id so the underlying
    # identity is one hover away even after we've prettified the label.
    tip_bits = [brand_name] if brand_name else []
    tip_bits.append(slug)
    if model_id:
        tip_bits.append(model_id)
    title = html.escape(" · ".join(tip_bits))
    return (
        f'<a href="{href}" class="cite cite-{provider}" title="{title}">'
        f'<span class="brand brand-{provider}">{_brand_glyph_html(provider)}</span>'
        f'<span class="cite-name">{html.escape(display)}</span>'
        f"{round_chip}"
        f"</a>"
    )


def _decorate_citations(
    html_text: str,
    slug_to_model: dict[str, str | None],
    bare_resolution: dict[str, str] | None = None,
) -> str:
    """Find `[slug]` citations in rendered HTML and replace each known slug
    with a coloured citation pill.

    A bracket-group is only decorated when at least one inner token matches
    a known slug — that way the rubric's literal `[brackets]` instruction
    and any non-citation bracketed text (e.g. `[Round 1]`) stay untouched.
    Unknown tokens inside a decorated group keep their original text. The
    surrounding `[...]` is dropped on a successful match since the pill is
    its own visual delimiter.

    `bare_resolution` maps bare slugs (`claude-haiku-2`) to their canonical
    round-suffixed counterpart (`claude-haiku-2.r3`) so cite-pill anchors
    land on a real `id=` even when the synthesiser omits the `.rN` suffix.
    """
    if not slug_to_model:
        return html_text
    bare_resolution = bare_resolution or {}

    def _decorate(match: re.Match[str]) -> str:
        inner = match.group(1)
        tokens = [t.strip() for t in inner.split(",")]
        if not any(t in slug_to_model for t in tokens):
            return match.group(0)
        out_tokens = [
            _cite_pill(t, slug_to_model.get(t), bare_resolution.get(t))
            if t in slug_to_model
            else html.escape(t)
            for t in tokens
        ]
        return ", ".join(out_tokens)

    return _CITATION_PATTERN.sub(_decorate, html_text)


def _slug_to_model(entries: list[dict[str, Any]]) -> dict[str, str | None]:
    """Build a slug → model_id map used to decorate citations in synthesis
    / arbiter text. Includes reconstructed earlier-round entries so r1
    citations work even though they aren't in the raw manifest.

    The map also carries bare-alias keys (`claude-haiku-2` alongside the
    full `claude-haiku-2.r3`) so a synthesiser that drops the `.rN` suffix
    still gets decorated. Use `_bare_resolution` to find the canonical
    anchor for those bare slugs.
    """
    direct: dict[str, str | None] = {e["slug"]: e.get("model_id") for e in entries}
    for bare, canonical in _bare_resolution(entries).items():
        direct.setdefault(bare, direct.get(canonical))
    return direct


def _bare_resolution(entries: list[dict[str, Any]]) -> dict[str, str]:
    """For each `<base>-<idx>.rN` slug, expose the bare `<base>-<idx>` as
    an alias resolving to the highest-round canonical slug. Explicit bare
    entries already in the manifest take precedence.
    """
    explicit = {e["slug"] for e in entries}
    latest_round: dict[str, int] = {}
    canonical: dict[str, str] = {}
    for e in entries:
        slug = e["slug"]
        m = _ROUND_SUFFIX.search(slug)
        if not m:
            continue
        bare = slug[: m.start()]
        if bare in explicit:
            continue
        round_n = int(m.group(1))
        if round_n > latest_round.get(bare, -1):
            latest_round[bare] = round_n
            canonical[bare] = slug
    return canonical


# ---- Run loading ------------------------------------------------------------


def _load_run_data(paths: artifacts.RunPaths) -> dict[str, Any]:
    if not paths.manifest_json.exists():
        raise FileNotFoundError(
            f"manifest.json missing for run {paths.run_id}; cannot render "
            "feed for a run that never produced a manifest"
        )
    manifest = json.loads(paths.manifest_json.read_text())
    prompt = paths.prompt_txt.read_text() if paths.prompt_txt.exists() else ""
    synth_path = paths.root / "synthesis.md"
    synth = synth_path.read_text() if synth_path.exists() else ""

    arbiters: list[dict[str, Any]] = []
    if paths.arbiters.exists():
        for p in sorted(paths.arbiters.glob("round-*.json")):
            try:
                arbiters.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("viewer: skipping malformed arbiter %s: %s", p, e)

    events: list[dict[str, Any]] = []
    log = paths.root / "_progress.log"
    if log.exists():
        for line in log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    manifest_entries = manifest.get("manifest", [])
    augmented = _augment_with_earlier_rounds(paths, manifest_entries, events)

    bodies: dict[str, str] = {}
    if paths.responses.exists():
        for body_file in paths.responses.glob("*.txt"):
            bodies[body_file.stem] = body_file.read_text()

    return {
        "manifest": manifest,
        "entries": augmented,
        "prompt": prompt,
        "synth": synth,
        "arbiters": arbiters,
        "events": events,
        "bodies": bodies,
        "cancelled": (paths.root / "CANCELLED").exists(),
    }


def _augment_with_earlier_rounds(
    paths: artifacts.RunPaths,
    manifest_entries: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For `refine` runs, `manifest.json` only carries the final round's
    panellists. Round-1 (and round-2) bodies + capsules exist on disk under
    `<slug>.r<n>` but aren't referenced by the manifest. Recover them so
    every round's panellists get cards, not just the last.

    The reconstructed entries carry whatever data can be recovered:
    status + latency from `_progress.log`, model_id + tokens from the raw
    response JSON, capsule from `capsules/<slug>.json`. Cost-per-panellist
    isn't recoverable for earlier rounds (refine doesn't persist it past
    the final round), so reconstructed entries set `cost_known=False`.
    Each reconstructed entry carries a `_reconstructed=True` marker so the
    UI can flag them.
    """
    by_slug = {e["slug"]: e for e in manifest_entries}
    if not paths.responses.exists():
        return list(manifest_entries)

    disk_slugs = {p.stem for p in paths.responses.glob("*.txt")}
    missing = disk_slugs - by_slug.keys()
    if not missing:
        return list(manifest_entries)

    # Map slug → most recent panellist_completed event (refine reuses slugs
    # only per round, so a single hit per slug is the norm).
    log_by_slug: dict[str, dict[str, Any]] = {}
    for ev in events:
        slug = ev.get("slug")
        kind = ev.get("kind")
        if not slug or kind not in ("panellist", "panellist_completed"):
            continue
        log_by_slug[slug] = ev

    extras: list[dict[str, Any]] = []
    for slug in sorted(missing):
        ev = log_by_slug.get(slug, {})
        # Status defaults to UNKNOWN if no progress log was kept (older runs);
        # the renderer treats UNKNOWN as a muted pill so it's visible.
        status = ev.get("status") or "UNKNOWN"
        latency_ms = ev.get("latency_ms")

        capsule_data: dict[str, Any] | None = None
        capsule_file = paths.capsules / f"{slug}.json"
        if capsule_file.exists():
            try:
                capsule_data = json.loads(capsule_file.read_text())
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("viewer: skipping malformed capsule %s: %s", capsule_file, e)

        model_id: str | None = None
        tokens_in: int | None = None
        tokens_out: int | None = None
        finish_reason: str | None = None
        raw_file = paths.responses / f"{slug}.json"
        if raw_file.exists():
            try:
                raw = json.loads(raw_file.read_text())
                model_id = raw.get("model")
                usage = raw.get("usage") or {}
                tokens_in = usage.get("prompt_tokens")
                tokens_out = usage.get("completion_tokens")
                choices = raw.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("viewer: skipping malformed raw response %s: %s", raw_file, e)

        # Refine reuses the model_id across rounds, so fall back to a
        # same-base-slug manifest entry's id when the raw json is missing
        # (rate-limited rounds have no .json file, only an empty .txt).
        if model_id is None:
            base = re.sub(r"\.r\d+$", "", slug)
            for e in manifest_entries:
                if re.sub(r"\.r\d+$", "", e["slug"]) == base:
                    model_id = e.get("model_id")
                    break

        extras.append(
            {
                "slug": slug,
                "model_id": model_id,
                "persona": None,
                "status": status,
                "finish_reason": finish_reason,
                "capsule": capsule_data,
                "confidence": (capsule_data or {}).get("confidence"),
                "resource_uri": paths.resource_uri(slug),
                "body_path": str(paths.responses / f"{slug}.txt"),
                "latency_ms": latency_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": None,
                "cost_known": False,
                "error": None,
                "_reconstructed": True,
            }
        )

    return list(manifest_entries) + extras


def _kind_of(arbiters: list, has_synth: bool) -> str:
    if arbiters:
        return "refine"
    if has_synth:
        return "consult"
    return "panel"


# ---- HTML ------------------------------------------------------------------

_CSS = """
  :root {
    --bg: #fafafa; --panel: #fff; --text: #111; --muted-text: #666;
    --border: #e5e5e5; --accent: #2c5cdb;
    --ok-bg: #e6f6e8; --ok-fg: #1f7a31;
    --warn-bg: #fdf3d6; --warn-fg: #8a5b00;
    --err-bg: #fbe3e3; --err-fg: #a01010;
    --muted-bg: #ececec; --muted-fg: #555;
    --code-bg: #f3f3f3;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --panel: #1c1f24; --text: #e8e8e8; --muted-text: #aaa;
      --border: #2c3038; --accent: #6f9bff;
      --ok-bg: #1c3a23; --ok-fg: #7ed792;
      --warn-bg: #3d3018; --warn-fg: #ffcb6b;
      --err-bg: #3d1c1c; --err-fg: #ff8585;
      --muted-bg: #262a30; --muted-fg: #c0c0c0;
      --code-bg: #0f1115;
    }
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text);
  }
  /* Jumped-to card briefly highlights so it's easy to spot. */
  .card:target {
    box-shadow: 0 0 0 2px var(--accent);
    transition: box-shadow 1.4s ease-out;
  }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 96px; }
  h1, h2, h3, h4 { line-height: 1.25; }
  h1 { font-size: 28px; margin: 0 0 4px; }
  h2 { font-size: 20px; margin: 32px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  h3 { font-size: 16px; margin: 16px 0 8px; }
  code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 13px; }
  pre { background: var(--code-bg); padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
  code { background: var(--code-bg); padding: 1px 5px; border-radius: 3px; }
  pre code { background: transparent; padding: 0; }
  a { color: var(--accent); }
  hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
  p { margin: 8px 0; }
  .meta { color: var(--muted-text); font-size: 13px; }
  .kind { text-transform: uppercase; letter-spacing: 0.08em; font-size: 11px; color: var(--muted-text); font-weight: 600; }
  .header-stats { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 8px; }
  .stat-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--muted-bg); padding: 4px 10px; border-radius: 6px;
    font-size: 12px; color: var(--muted-text);
  }
  .stat-chip b { color: var(--text); font-weight: 600; }
  .status-counts { margin: 8px 0 0; display: flex; flex-wrap: wrap; gap: 6px; }
  .pill {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; vertical-align: 1px;
  }
  .pill-ok { background: var(--ok-bg); color: var(--ok-fg); }
  .pill-warn { background: var(--warn-bg); color: var(--warn-fg); }
  .pill-err { background: var(--err-bg); color: var(--err-fg); }
  .pill-muted { background: var(--muted-bg); color: var(--muted-fg); }
  details {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; margin: 8px 0;
  }
  details > summary { cursor: pointer; font-weight: 500; }
  details[open] { padding-bottom: 14px; }
  details > summary::marker { color: var(--muted-text); }
  .prompt-box { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
  .prompt-box pre { white-space: pre-wrap; background: transparent; padding: 0; }
  /* Synthesis is the answer the user opens the page for — accent-tinted
     card and a left rule so it visually outranks the panellist grid. */
  .synth-card {
    background: color-mix(in srgb, var(--accent) 6%, var(--panel));
    border: 1px solid color-mix(in srgb, var(--accent) 22%, var(--border));
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 14px 22px 18px;
    font-size: 15.5px; line-height: 1.6;
    margin: 8px 0 4px;
  }
  .synth-card > :first-child { margin-top: 0; }
  .synth-card > :last-child { margin-bottom: 0; }
  /* Markdown the synth model emits has its own h1/h2 — downgrade them so
     they nest inside the section h2 instead of competing with it. */
  .synth-card h1 {
    font-size: 18px; margin: 18px 0 8px;
    padding-bottom: 4px; border-bottom: 1px solid var(--border);
  }
  .synth-card h2 {
    font-size: 16px; margin: 14px 0 6px;
    padding: 0; border: none;
  }
  .synth-card h3 { font-size: 14px; margin: 12px 0 4px; }
  .synth-card ul, .synth-card ol { padding-left: 22px; margin: 6px 0; }
  .synth-card li { margin: 3px 0; }
  /* Secondary synthesis sections (risks, next steps, caveats, appendix)
     collapse-by-default — the global `details` panel border would compete
     with the synth card's chrome, so neutralise it and let the heading
     tag inside <summary> carry the visual weight. */
  .synth-fold {
    background: transparent; border: none; padding: 0;
    margin: 14px 0 0;
  }
  .synth-fold[open] { padding-bottom: 0; }
  .synth-fold > summary {
    cursor: pointer; padding: 0; margin: 0;
    list-style-position: outside;
  }
  .synth-fold > summary::marker { color: var(--muted-text); font-size: 11px; }
  .synth-fold > summary > h1,
  .synth-fold > summary > h2 {
    display: inline; padding: 0; border: none; margin: 0;
  }
  /* When a section is folded, show the heading's bottom rule on the
     summary line so the page still reads as separated sections. */
  .synth-fold:not([open]) > summary > h1,
  .synth-fold:not([open]) > summary > h2 {
    color: var(--muted-text);
  }
  .panel-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap: 16px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .card-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 4px; }
  /* Model badge + name is the card's heading now that the raw slug is
     gone — bump weight and size so it reads as a title, not metadata. */
  .card-head .model-tag { font-size: 14.5px; }
  .card-head .model-tag .model-name { font-weight: 600; }
  .model-tag {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; color: var(--text); white-space: nowrap;
  }
  .model-tag .model-name { font-weight: 500; }
  .brand {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; flex-shrink: 0;
    border-radius: 5px; color: #fff;
  }
  .brand-icon { width: 12px; height: 12px; fill: currentColor; display: block; }
  /* Each provider sets `--rgb` once; the global `.brand` and `.cite`
     rules below render the solid badge and the tinted citation pill from
     the same colour, halving the per-provider rule count. */
  .brand-anthropic,  .cite-anthropic  { --rgb: 217, 119, 87; }
  .brand-openai,     .cite-openai     { --rgb: 16, 163, 127; }
  .brand-google,     .cite-google     { --rgb: 66, 133, 244; }
  .brand-xai,        .cite-xai        { --rgb: 44, 44, 44; }
  .brand-deepseek,   .cite-deepseek   { --rgb: 124, 58, 237; }
  .brand-mistral,    .cite-mistral    { --rgb: 250, 62, 62; }
  .brand-meta,       .cite-meta       { --rgb: 24, 119, 242; }
  .brand-perplexity, .cite-perplexity { --rgb: 32, 164, 181; }
  .brand-qwen,       .cite-qwen       { --rgb: 212, 57, 28; }
  .brand-moonshot,   .cite-moonshot   { --rgb: 179, 101, 240; }
  .brand-zhipu,      .cite-zhipu      { --rgb: 37, 71, 168; }
  .brand-xiaomi,     .cite-xiaomi     { --rgb: 249, 115, 22; }
  .brand-other,      .cite-other      { --rgb: 138, 138, 138; }
  .brand { background: rgb(var(--rgb)); }
  .recon-badge {
    display: inline-block; padding: 1px 6px; border-radius: 999px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--muted-fg);
    background: var(--muted-bg);
  }
  .cite {
    display: inline-flex; align-items: center; gap: 4px;
    white-space: nowrap; vertical-align: 1px;
    font-family: inherit;
    padding: 1px 7px 1px 4px;
    border-radius: 999px;
    text-decoration: none; color: inherit;
  }
  .cite:hover { filter: brightness(0.92); }
  @media (prefers-color-scheme: dark) {
    .cite:hover { filter: brightness(1.15); }
  }
  .cite-name { font-weight: 500; }
  .cite .brand {
    width: 14px; height: 14px; border-radius: 4px;
  }
  .cite .brand .brand-icon { width: 9px; height: 9px; }
  /* Tinted background — provider's `--rgb` at low opacity. Bumps in
     dark mode where pale tints get lost. */
  .cite { background: rgba(var(--rgb), 0.16); }
  @media (prefers-color-scheme: dark) {
    .cite { background: rgba(var(--rgb), 0.26); }
    /* xAI's near-black brand reads as a dead-space hole on dark panels —
       swap to a light grey + dark text so the badge still pops. */
    .brand-xai, .cite-xai { --rgb: 217, 217, 217; }
    .brand-xai { color: #111; }
  }
  .cite-round {
    display: inline-block; padding: 0 5px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.08); color: var(--muted-fg);
    font-size: 10px; font-weight: 600;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    letter-spacing: 0.03em;
  }
  @media (prefers-color-scheme: dark) {
    .cite-round { background: rgba(255, 255, 255, 0.14); }
  }
  .card-stats {
    display: flex; flex-wrap: wrap; gap: 4px 14px;
    font-size: 12px; color: var(--muted-text); margin: 4px 0 10px;
  }
  .card-stats b { color: var(--text); font-weight: 500; }
  .capsule-row { font-size: 13px; margin: 6px 0; }
  .capsule-row .label {
    color: var(--muted-text); font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em; margin-right: 6px;
  }
  .capsule-row ul { margin: 4px 0 0; padding-left: 22px; }
  .capsule-row li { margin: 2px 0; }
  .capsule {
    margin-top: 10px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  .cap-position {
    font-size: 14px;
    font-weight: 500;
    line-height: 1.5;
    margin: 0 0 12px;
  }
  .cap-rec {
    display: block;
    background: rgba(44, 92, 219, 0.07);
    border-left: 3px solid var(--accent);
    padding: 9px 12px 10px;
    border-radius: 0 6px 6px 0;
    margin: 0 0 14px;
    font-size: 13px;
    line-height: 1.5;
  }
  @media (prefers-color-scheme: dark) {
    .cap-rec { background: rgba(111, 155, 255, 0.12); }
  }
  .cap-rec-label {
    display: block;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 3px;
  }
  .cap-rec-body { display: block; }
  /* cap-list is a <details>, but unlike the global card-style details we
     don't want a panel border or padding — these are inline summary
     blocks inside a capsule. */
  .cap-list {
    margin: 8px 0; background: transparent; border: none; padding: 0;
  }
  .cap-list[open] { padding-bottom: 0; }
  .cap-list-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted-text);
    cursor: pointer;
  }
  .cap-list-label::marker { font-size: 9px; color: var(--muted-text); }
  .cap-list > ul { margin: 6px 0 0; padding-left: 22px; font-size: 13px; line-height: 1.5; }
  .cap-list > ul > li { margin: 3px 0; }
  .cap-list-unique > ul { color: var(--text); }
  .cap-list-unique > ul > li::marker { color: var(--accent); }
  .cap-list-caveat {
    background: rgba(138, 91, 0, 0.06);
    border-radius: 6px;
    padding: 6px 10px;
  }
  .cap-list-caveat[open] { padding-bottom: 8px; }
  @media (prefers-color-scheme: dark) {
    .cap-list-caveat { background: rgba(255, 203, 107, 0.08); }
  }
  .cap-list-caveat .cap-list-label { color: var(--warn-fg); }
  .cap-list-caveat .cap-list-label::marker { color: var(--warn-fg); }
  .cap-list-caveat > ul { color: var(--text); }
  .cap-cites {
    margin: 10px 0;
    display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center;
  }
  .cap-cites .cap-list-label {
    margin-bottom: 0; margin-right: 4px;
  }
  .cap-cites-agree .cap-list-label { color: var(--ok-fg); }
  .cap-cites-disagree .cap-list-label { color: var(--err-fg); }
  .round-header {
    font-size: 12px; color: var(--muted-text); text-transform: uppercase;
    letter-spacing: 0.06em; margin: 24px 0 8px; padding-bottom: 4px;
    border-bottom: 1px dashed var(--border);
  }
  /* Refine rounds read as a sequence — a vertical rail with a dot per
     round makes the iteration visible at a glance. */
  .arbiter-progression { position: relative; padding-left: 28px; margin-top: 8px; }
  .arbiter-progression::before {
    content: ""; position: absolute; left: 8px; top: 24px; bottom: 24px;
    width: 2px; background: var(--border);
  }
  .arbiter-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin: 0 0 14px;
    position: relative;
  }
  .arbiter-card:last-child { margin-bottom: 0; }
  .arbiter-progression .arbiter-card::before {
    content: ""; position: absolute; left: -27px; top: 20px;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--accent); border: 3px solid var(--bg);
  }
  .arbiter-head { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
  .arbiter-head .round-num { font-weight: 600; font-size: 15px; }
  .score-bar {
    display: inline-block; width: 140px; height: 8px;
    background: var(--muted-bg); border-radius: 4px; overflow: hidden;
  }
  .score-bar > i { display: block; height: 100%; background: var(--accent); }
  /* Score tone propagates from the arbiter card to both the bar fill and
     the rail dot — a glance at the rail then tells the refine story
     (red round-1 → amber round-2 → green round-3). */
  .arbiter-card.score-ok   .score-bar > i { background: var(--ok-fg); }
  .arbiter-card.score-warn .score-bar > i { background: var(--warn-fg); }
  .arbiter-card.score-err  .score-bar > i { background: var(--err-fg); }
  .arbiter-progression .arbiter-card.score-ok::before   { background: var(--ok-fg); }
  .arbiter-progression .arbiter-card.score-warn::before { background: var(--warn-fg); }
  .arbiter-progression .arbiter-card.score-err::before  { background: var(--err-fg); }
  /* Timeline is debug detail for most readers; the collapsible wrapper
     reuses the same neutralised-details pattern as `.synth-fold` and
     `.cap-list` so the timeline header doesn't read as a card. */
  .timeline-fold {
    background: transparent; border: none; padding: 0; margin: 0;
  }
  .timeline-fold[open] { padding-bottom: 0; }
  .timeline-fold > summary {
    cursor: pointer; color: var(--muted-text); font-size: 13px;
  }
  .timeline-fold[open] > summary { margin-bottom: 8px; }
  .timeline { list-style: none; padding: 0; margin: 0; font-size: 13px; }
  .timeline li {
    display: grid; grid-template-columns: 70px 140px minmax(120px, 1fr) auto auto;
    gap: 12px; padding: 5px 0; border-bottom: 1px dashed var(--border);
    align-items: baseline;
  }
  .timeline .t { color: var(--muted-text); font-variant-numeric: tabular-nums; }
  .timeline .k { color: var(--muted-text); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .timeline .slug { font-weight: 500; }
  .timeline .pills { display: flex; gap: 6px; }
  .timeline .lat { color: var(--muted-text); font-variant-numeric: tabular-nums; }
  .error-msg {
    color: var(--err-fg); font-size: 12px;
    font-family: "SF Mono", Menlo, monospace;
    background: var(--err-bg); padding: 8px 10px;
    border-radius: 6px; margin: 6px 0; white-space: pre-wrap;
    overflow-x: auto;
  }
  /* Informational annotation on a successful call (e.g. auto-trim). Same
     shape as .error-msg but neutral colours — the call succeeded; we're
     just surfacing context that affected this panellist's view. */
  .note-msg {
    color: var(--muted-fg); font-size: 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--muted-bg); padding: 6px 10px;
    border-radius: 6px; margin: 6px 0; white-space: pre-wrap;
    overflow-x: auto;
  }
  .footer { color: var(--muted-text); font-size: 12px; margin-top: 48px; padding-top: 12px; border-top: 1px solid var(--border); }
  .body-pre {
    white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px; background: transparent; padding: 0; line-height: 1.55;
  }
  .partial-banner {
    background: var(--warn-bg); color: var(--warn-fg);
    padding: 10px 14px; border-radius: 8px; margin: 12px 0; font-size: 13px;
  }
  .cancelled-banner {
    background: var(--err-bg); color: var(--err-fg);
    padding: 10px 14px; border-radius: 8px; margin: 12px 0; font-size: 13px;
  }
  @media print {
    /* Force light palette regardless of OS preference. Prevents browsers
       on dark-mode systems from printing a black page on a black page. */
    :root {
      --bg: #fff; --panel: #fff; --text: #000; --muted-text: #444;
      --border: #ccc; --muted-bg: #eee; --muted-fg: #333; --code-bg: #f4f4f4;
    }
    body { background: #fff; color: #000; }
    main { max-width: none; padding: 0 6px; }
    /* Keep cards and arbiters intact across page boundaries. */
    .card, .arbiter-card, .synth-card, .prompt-box {
      break-inside: avoid; box-shadow: none;
    }
    h2 { break-after: avoid; }
    /* Debug noise the user doesn't want in a saved archive. */
    .timeline, .footer { display: none; }
    /* Preserve the brand/cite tints — browsers strip background colour
       on print by default. */
    .brand, .cite, .pill, .stat-chip, .synth-card, .cap-list-caveat {
      -webkit-print-color-adjust: exact; print-color-adjust: exact;
    }
    /* Citation anchors don't navigate on paper — keep them visible as
       coloured pills but drop the link underline. */
    a.cite { color: inherit; text-decoration: none; }
  }
"""


def _header(run_id: str, kind: str, manifest: dict, cancelled: bool) -> str:
    entries = manifest.get("manifest", [])
    counts = Counter(e.get("status", "UNKNOWN") for e in entries)
    counts_html = "".join(
        _pill(f"{n} {status}", _status_tone(status)) for status, n in sorted(counts.items())
    )
    cost = manifest.get("cost_usd")
    cost_known = manifest.get("cost_known", True)
    wall_ms = manifest.get("wall_ms", 0)
    partial = manifest.get("partial", False)
    partial_reason = manifest.get("partial_reason")
    blinded = manifest.get("blinded", False)
    synth_model = manifest.get("synthesiser")

    stats_pieces = [
        f'<span class="stat-chip"><b>{len(entries)}</b> panellists</span>',
        f'<span class="stat-chip">cost <b>{html.escape(_fmt_cost(cost, cost_known))}</b></span>',
        f'<span class="stat-chip">wall <b>{html.escape(_fmt_ms(wall_ms))}</b></span>',
    ]
    if blinded:
        stats_pieces.append('<span class="stat-chip"><b>blinded</b></span>')
    if synth_model:
        stats_pieces.append(f'<span class="stat-chip">synth {_model_badge_html(synth_model)}</span>')

    banner = ""
    if cancelled:
        banner += '<div class="cancelled-banner">This run was cancelled before completion.</div>'
    if partial and partial_reason:
        banner += f'<div class="partial-banner">Partial run: {html.escape(partial_reason)}</div>'

    return f"""
    <header>
      <div class="kind">{html.escape(kind)} run</div>
      <h1>{html.escape(run_id)}</h1>
      <div class="header-stats">{"".join(stats_pieces)}</div>
      <div class="status-counts">{counts_html}</div>
      {banner}
    </header>
    """


def _section_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    # Collapse internal whitespace for the preview so the summary doesn't
    # include awkward blank lines; the full prompt inside still preserves
    # original formatting.
    preview_src = re.sub(r"\s+", " ", prompt).strip()
    preview = preview_src[:160]
    suffix = "…" if len(preview_src) > 160 else ""
    char_count = f' <span class="meta">· {len(prompt)} chars</span>'
    return f"""
    <section>
      <h2>Prompt</h2>
      <details>
        <summary>{html.escape(preview)}{suffix}{char_count}</summary>
        <div class="prompt-box"><pre>{html.escape(prompt)}</pre></div>
      </details>
    </section>
    """


# Section headings that should fold by default — secondary appendices the
# reader doesn't need to scroll past to see the headline answer. Matched
# generically (case-insensitive substring) so the same logic works across
# the consensus, critique, code-review, and research-brief rubrics.
_SYNTH_FOLD_KEYWORDS = (
    "risk",
    "next step",
    "caveat",
    "appendix",
    "alternatives considered",
    "footnote",
    "further reading",
)

_SYNTH_HEADING_RE = re.compile(r"<(h[12])>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _fold_synth_sections(rendered: str) -> str:
    """Split the rendered synth on `<h1>`/`<h2>` boundaries; wrap secondary
    sections whose heading text matches `_SYNTH_FOLD_KEYWORDS` in a
    collapsible `<details>` block. The first heading always stays open so
    the headline answer is visible without a click.
    """
    # Pre-scan for headings; if there are none, the synth is flat prose and
    # there's nothing useful to fold — bail before the split overhead.
    headings = list(_SYNTH_HEADING_RE.finditer(rendered))
    if not headings:
        return rendered
    out: list[str] = []
    cursor = 0
    for i, m in enumerate(headings):
        # Everything between the previous heading's end and this heading's
        # start belongs to the previous section's body.
        if cursor < m.start():
            out.append(rendered[cursor : m.start()])
        # Determine where this section's body ends — at the next heading's
        # start, or at end-of-string.
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(rendered)
        heading_html = m.group(0)
        body_html = rendered[m.end() : body_end]
        heading_text = _HTML_TAG_RE.sub("", m.group(2)).strip().lower()
        is_fold = i > 0 and any(kw in heading_text for kw in _SYNTH_FOLD_KEYWORDS)
        if is_fold:
            # Keep the original heading tag inside `<summary>` so its
            # styling carries over; the global `.synth-fold > summary` CSS
            # neutralises the summary's own padding so the heading sits
            # flush.
            out.append(f'<details class="synth-fold"><summary>{heading_html}</summary>{body_html}</details>')
        else:
            out.append(heading_html)
            out.append(body_html)
        cursor = body_end
    if cursor < len(rendered):
        out.append(rendered[cursor:])
    return "".join(out)


def _section_synth(
    synth_md: str,
    slug_to_model: dict[str, str | None],
    bare_resolution: dict[str, str] | None = None,
) -> str:
    if not synth_md.strip():
        return ""
    rendered = render_markdown(synth_md)
    rendered = _decorate_citations(rendered, slug_to_model, bare_resolution)
    rendered = _fold_synth_sections(rendered)
    return f"""
    <section>
      <h2>Synthesis</h2>
      <div class="synth-card">{rendered}</div>
    </section>
    """


def _section_arbiters(
    arbiters: list[dict[str, Any]],
    slug_to_model: dict[str, str | None],
    bare_resolution: dict[str, str] | None = None,
) -> str:
    if not arbiters:
        return ""

    def _cite(text: str) -> str:
        # Arbiter prose is plain text — escape, then decorate slug citations
        # so `[slug]` references get the same badge treatment as synthesis.
        return _decorate_citations(html.escape(text), slug_to_model, bare_resolution)

    cards: list[str] = []
    for v in arbiters:
        score = float(v.get("score", 0) or 0)
        bar_pct = max(0, min(100, int(score * 100)))
        score_tone = "ok" if score >= 0.85 else ("warn" if score >= 0.6 else "err")
        gaps = v.get("gaps") or []
        gaps_html = (
            ("<ul>" + "".join(f"<li>{_cite(g)}</li>" for g in gaps) + "</ul>")
            if gaps
            else '<p class="meta">No gaps recorded.</p>'
        )
        focus = v.get("next_round_focus") or ""
        reasoning = v.get("reasoning") or ""
        cost = v.get("cost_usd")
        cost_known = v.get("cost_known", True)
        err = v.get("error")
        parsed_ok = v.get("parsed_ok", True)
        meta_bits = [f"<span>cost <b>{html.escape(_fmt_cost(cost, cost_known))}</b></span>"]
        if not parsed_ok:
            meta_bits.append(_pill("PARSE FAILED", "err"))
        # `score-{tone}` class lets CSS tint the bar fill and the
        # progression rail dot to match the score band.
        cards.append(f"""
        <div class="arbiter-card score-{score_tone}">
          <div class="arbiter-head">
            <span class="round-num">Round {int(v.get("round", 0))}</span>
            {_pill(f"score {score:.2f}", score_tone)}
            <span class="score-bar"><i style="width: {bar_pct}%"></i></span>
            <span class="card-stats">{"".join(meta_bits)}</span>
          </div>
          <div class="capsule-row"><span class="label">Gaps</span>{gaps_html}</div>
          {f'<div class="capsule-row"><span class="label">Next focus</span>{_cite(focus)}</div>' if focus else ""}
          {f'<div class="capsule-row"><span class="label">Reasoning</span>{_cite(reasoning)}</div>' if reasoning else ""}
          {f'<div class="error-msg">{html.escape(err)}</div>' if err else ""}
        </div>
        """)
    return f"""
    <section>
      <h2>Arbiter rounds</h2>
      <div class="arbiter-progression">{"".join(cards)}</div>
    </section>
    """


def _capsule_block(
    capsule: dict | None,
    slug_to_model: dict[str, str | None],
    bare_resolution: dict[str, str] | None = None,
) -> str:
    """Render the structured capsule fields with visual hierarchy:

    - position is the headline (no label, larger weight)
    - recommendation gets an accent-tinted callout
    - key_points / unique_claims / caveats are labelled bullet lists, with
      caveats tinted warn so hedges read as hedges at a glance
    - agrees_with / disagrees_with are rendered as cite pills since they
      reference panellist slugs — the same pill the synthesis decorator
      uses, so the inter-panellist relationships read consistently
      regardless of where they show up in the page.
    """
    if not capsule:
        return ""
    parts: list[str] = []

    position = (capsule.get("position") or "").strip()
    if position:
        parts.append(f'<div class="cap-position">{html.escape(position)}</div>')

    recommendation = (capsule.get("recommendation") or "").strip()
    if recommendation:
        parts.append(
            '<div class="cap-rec">'
            '<span class="cap-rec-label">Recommendation</span>'
            f'<span class="cap-rec-body">{html.escape(recommendation)}</span>'
            "</div>"
        )

    def _list_section(label: str, items: list[str], variant: str = "") -> str:
        if not items:
            return ""
        lis = "".join(f"<li>{html.escape(it)}</li>" for it in items)
        cls = f"cap-list cap-list-{variant}" if variant else "cap-list"
        # Collapsed by default — keeps panellist cards scannable. The count
        # in the summary tells the user how much they'd expand.
        return (
            f'<details class="{cls}">'
            f'<summary class="cap-list-label">{html.escape(label)} ({len(items)})</summary>'
            f"<ul>{lis}</ul>"
            f"</details>"
        )

    parts.append(_list_section("Key points", capsule.get("key_points") or []))
    parts.append(_list_section("Unique claims", capsule.get("unique_claims") or [], "unique"))
    parts.append(_list_section("Caveats", capsule.get("caveats") or [], "caveat"))

    resolution = bare_resolution or {}

    def _cite_row(label: str, slugs: list[str], variant: str) -> str:
        if not slugs:
            return ""
        pills = " ".join(_cite_pill(s, slug_to_model.get(s), resolution.get(s)) for s in slugs)
        return (
            f'<div class="cap-cites cap-cites-{variant}">'
            f'<span class="cap-list-label">{html.escape(label)}</span>'
            f"{pills}"
            f"</div>"
        )

    parts.append(_cite_row("Agrees with", capsule.get("agrees_with") or [], "agree"))
    parts.append(_cite_row("Disagrees with", capsule.get("disagrees_with") or [], "disagree"))

    body = "".join(p for p in parts if p)
    if not body:
        return ""
    return f'<div class="capsule">{body}</div>'


def _panellist_card(
    entry: dict,
    body: str,
    slug_to_model: dict[str, str | None],
    bare_resolution: dict[str, str] | None = None,
) -> str:
    slug = entry["slug"]
    status = entry.get("status") or "UNKNOWN"
    model_id = entry.get("model_id")
    persona = entry.get("persona")
    latency = entry.get("latency_ms")
    cost = entry.get("cost_usd")
    cost_known = entry.get("cost_known", True)
    confidence = entry.get("confidence")
    tokens_in = entry.get("tokens_in")
    tokens_out = entry.get("tokens_out")
    error = entry.get("error")
    note = entry.get("note")
    capsule = entry.get("capsule")
    reconstructed = entry.get("_reconstructed", False)

    stats: list[str] = [
        f"<span>{html.escape(_fmt_ms(latency))}</span>",
    ]
    # Reconstructed entries have no recoverable cost — omit rather than
    # rendering a misleading "?" pricing-unknown sentinel.
    if not reconstructed or cost is not None:
        stats.append(f"<span>{html.escape(_fmt_cost(cost, cost_known))}</span>")
    if confidence is not None:
        stats.append(f"<span>conf <b>{confidence:.2f}</b></span>")
    if tokens_in or tokens_out:
        stats.append(f"<span>tok <b>{tokens_in or 0}→{tokens_out or 0}</b></span>")
    if persona:
        stats.append(f"<span>persona <b>{html.escape(persona)}</b></span>")

    err_block = f'<div class="error-msg">{html.escape(error)}</div>' if error else ""
    note_block = f'<div class="note-msg">{html.escape(note)}</div>' if note else ""
    cap_block = _capsule_block(capsule, slug_to_model, bare_resolution)
    recon_tag = (
        '<span class="recon-badge" title="Reconstructed from on-disk artifacts '
        "(refine's manifest.json only carries the final round)\">reconstructed</span>"
        if reconstructed
        else ""
    )

    if body:
        body_block = f"""
        <details>
          <summary>Full body ({len(body)} chars)</summary>
          <pre class="body-pre">{html.escape(body)}</pre>
        </details>
        """
    else:
        body_block = '<p class="meta">No body recorded.</p>' if status != "OK" else ""

    # `slug` survives only as the article's id (citation jump target) and
    # the title tooltip — the heading is now the humanised model name so
    # `claude-haiku-2.r1 · Claude Haiku 4.5` no longer reads as a stutter.
    return f"""
    <article class="card" id="card-{html.escape(slug, quote=True)}" title="{html.escape(slug)}">
      <div class="card-head">
        {_model_badge_html(model_id, slug)}
        {_pill(status, _status_tone(status))}
        {recon_tag}
      </div>
      <div class="card-stats">{"".join(stats)}</div>
      {err_block}
      {note_block}
      {cap_block}
      {body_block}
    </article>
    """


def _section_panellists(
    entries: list[dict],
    bodies: dict[str, str],
    slug_to_model: dict[str, str | None],
    bare_resolution: dict[str, str] | None = None,
) -> str:
    if not entries:
        return '<section><h2>Panellists</h2><p class="meta">No panellists recorded.</p></section>'

    # Bucket by round if any slug carries an .rN suffix; otherwise one flat group.
    groups: dict[int | None, list[dict]] = {}
    for entry in entries:
        groups.setdefault(_round_of(entry["slug"]), []).append(entry)
    multi_round = any(k is not None for k in groups) and len(groups) > 1

    parts: list[str] = ["<section><h2>Panellists</h2>"]
    for key in sorted(groups, key=lambda k: (k is None, k or 0)):
        if multi_round:
            label = f"Round {key}" if key is not None else "Initial"
            parts.append(f'<div class="round-header">{html.escape(label)}</div>')
        cards = [
            _panellist_card(e, bodies.get(e["slug"], ""), slug_to_model, bare_resolution) for e in groups[key]
        ]
        parts.append(f'<div class="panel-grid">{"".join(cards)}</div>')
    parts.append("</section>")
    return "".join(parts)


def _section_timeline(events: list[dict[str, Any]]) -> str:
    if not events:
        return ""
    # Compute relative offsets from the first parseable timestamp.
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    for e in events:
        first_ts = _parse_ts(e.get("ts"))
        if first_ts is not None:
            break
    for e in reversed(events):
        last_ts = _parse_ts(e.get("ts"))
        if last_ts is not None:
            break
    span_disp = ""
    if first_ts and last_ts and last_ts >= first_ts:
        delta_ms = int((last_ts - first_ts).total_seconds() * 1000)
        span_disp = f" · spans {_fmt_ms(delta_ms)}"

    rows: list[str] = []
    for e in events:
        kind = str(e.get("kind", "")) or "event"
        ts = _parse_ts(e.get("ts"))
        if ts and first_ts:
            delta = (ts - first_ts).total_seconds()
            t_disp = f"+{delta:.1f}s" if delta >= 0 else f"{delta:.1f}s"
        elif ts:
            t_disp = ts.strftime("%H:%M:%S")
        else:
            t_disp = "—"
        slug = str(e.get("slug", ""))
        status = e.get("status")
        latency_ms = e.get("latency_ms")
        score = e.get("score")
        round_num = e.get("round")
        step = e.get("step")

        # Right-hand detail varies by kind. Keep it terse — the timeline is for
        # scanning, not for full inspection (cards already have the depth).
        detail_bits: list[str] = []
        if status:
            detail_bits.append(_pill(str(status), _status_tone(str(status))))
        if score is not None:
            detail_bits.append(_pill(f"score {float(score):.2f}", "muted"))
        if round_num is not None and "round" in kind:
            detail_bits.append(_pill(f"r{round_num}", "muted"))
        if step is not None:
            detail_bits.append(_pill(f"step {step}", "muted"))
        lat_disp = _fmt_ms(latency_ms) if isinstance(latency_ms, int) else ""

        rows.append(f"""
        <li>
          <span class="t" title="{html.escape(e.get("ts", "") or "")}">{html.escape(t_disp)}</span>
          <span class="k">{html.escape(kind)}</span>
          <span class="slug">{html.escape(slug)}</span>
          <span class="pills">{"".join(detail_bits)}</span>
          <span class="lat">{html.escape(lat_disp)}</span>
        </li>
        """)
    summary = f"{len(events)} events{span_disp}"
    return f"""
    <section>
      <h2>Timeline</h2>
      <details class="timeline-fold">
        <summary>{html.escape(summary)}</summary>
        <ul class="timeline">{"".join(rows)}</ul>
      </details>
    </section>
    """


def _section_footer(paths: artifacts.RunPaths) -> str:
    return f"""
    <div class="footer">
      Run dir: <code>{html.escape(str(paths.root))}</code>
    </div>
    """


def _build_html(paths: artifacts.RunPaths, data: dict[str, Any]) -> str:
    manifest = data["manifest"]
    # `entries` carries the augmented list (final-round manifest entries + any
    # reconstructed earlier-round entries from disk); `manifest['manifest']`
    # is the raw last-round-only list. Header status counts use the raw list
    # so totals match what the user would see in the run's own manifest;
    # the panel section uses the augmented list so all rounds are visible.
    entries = data["entries"]
    slug_to_model = _slug_to_model(entries)
    bare_resolution = _bare_resolution(entries)
    kind = _kind_of(data["arbiters"], bool(data["synth"].strip()))
    title = f"consult {paths.run_id}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_CSS}</style>
</head>
<body>
{_brand_sprite_html()}
<main>
{_header(paths.run_id, kind, manifest, data["cancelled"])}
{_section_synth(data["synth"], slug_to_model, bare_resolution)}
{_section_prompt(data["prompt"])}
{_section_arbiters(data["arbiters"], slug_to_model, bare_resolution)}
{_section_panellists(entries, data["bodies"], slug_to_model, bare_resolution)}
{_section_timeline(data["events"])}
{_section_footer(paths)}
</main>
</body>
</html>
"""


def render_run(run_id: str) -> Path:
    """Read every artifact under the run dir and write `feed.html` next to
    them. Returns the absolute path of the output. Overwrites any existing
    `feed.html` from a prior render — the feed is a pure derivation of the
    source artifacts so regenerating is always safe.
    """
    paths = artifacts.load_run(run_id)
    data = _load_run_data(paths)
    out = paths.root / "feed.html"
    out.write_text(_build_html(paths, data))
    return out


# ---- CLI -------------------------------------------------------------------


def cli() -> None:
    """Console entry point.

    Usage:
      consult-view <run_id>          generate feed.html, print its path
      consult-view <run_id> --open   also open it in the default browser
    """
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="consult-view",
        description="Render a consult run as a self-contained HTML page.",
    )
    parser.add_argument("--version", action="version", version=f"consult-view {__version__}")
    parser.add_argument("run_id", help="A run_id under ~/.consult/runs/")
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the generated file in the default browser after writing.",
    )
    args = parser.parse_args()

    try:
        out = render_run(args.run_id)
    except FileNotFoundError as e:
        raise SystemExit(f"consult-view: {e}") from e

    print(str(out))
    if args.open_browser:
        webbrowser.open(out.as_uri())
