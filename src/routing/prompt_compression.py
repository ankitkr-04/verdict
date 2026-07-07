"""Prompt minimization for escalation calls — every character sent remote is scored.

Compression is deliberately conservative: layout-preserving (code indentation survives),
content-preserving (quoted passages are never paraphrased). The big savings come from
elsewhere: no few-shots, near-empty shared system message, tight max_tokens.
"""

from __future__ import annotations

import re

_LEADING_COURTESY_RE = re.compile(
    r"^(?:hi|hello|hey|please|kindly|greetings)[,!. ]+\s*", re.IGNORECASE
)
_TRAILING_COURTESY_RE = re.compile(
    r"\s*(?:thanks(?: in advance)?|thank you)[,!. ]*$", re.IGNORECASE
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_INLINE_WS_RE = re.compile(r"(?<=\S)[ \t]{2,}")


def compress_prompt(prompt: str) -> str:
    """Trim courtesies and squeeze whitespace without touching per-line indentation."""
    lines = [_INLINE_WS_RE.sub(" ", line.rstrip()) for line in prompt.strip().splitlines()]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _LEADING_COURTESY_RE.sub("", text)
    text = _TRAILING_COURTESY_RE.sub("", text)
    return text.strip()


def build_escalation_user(prompt: str, instruction: str) -> str:
    """Compressed task + the category's terse output instruction."""
    compressed = compress_prompt(prompt)
    if not instruction:
        return compressed
    return f"{compressed}\n\n{instruction}"
