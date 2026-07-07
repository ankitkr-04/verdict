"""Answer-extraction and text-measurement utilities used by solvers and verifiers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_FINAL_RE = re.compile(r"^\s*FINAL\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z0-9'-]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for with by from as and or "
    "but if then than that this these those it its it's not no do does did done has have "
    "had can could will would should may might about into over under between".split()
)

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a single": 1,
}


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by thinking-mode models."""
    return _THINK_RE.sub("", text).strip()


def extract_code_block(text: str) -> str | None:
    """Return the last fenced code block (models put the final version last)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return None


def extract_code_or_all(text: str) -> str:
    """Fenced block if present, else the raw text (model skipped the fences)."""
    return extract_code_block(text) or text.strip()


def extract_final(text: str) -> str | None:
    """Last 'FINAL: ...' line, if any."""
    matches = _FINAL_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_json_array(text: str) -> list | None:
    """Parse the first JSON array found (fenced or inline), else None."""
    candidate = extract_code_block(text) or text
    start = candidate.find("[")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(candidate[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def extract_label(text: str, labels: list[str]) -> str | None:
    """Find a classification label: prefer exact first-line word, else first occurrence."""
    clean = strip_thinking(text).strip()
    if not clean:
        return None
    first_line = clean.splitlines()[0].strip().strip(".:*_ ").lower()
    for label in labels:
        if first_line == label.lower():
            return label
    lowered = clean.lower()
    hits = [(lowered.find(lbl.lower()), lbl) for lbl in labels if lbl.lower() in lowered]
    return min(hits)[1] if hits else None


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line))


def normalize_answer(text: str) -> str:
    """Canonical form for majority voting / agreement checks."""
    text = strip_thinking(text).lower().strip()
    text = re.sub(r"[^\w\s.%-]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def extract_numbers(text: str) -> list[str]:
    """Numbers normalized: commas stripped, trailing zeros trimmed ('1,200.50' -> '1200.5')."""
    out = []
    for raw in _NUMBER_RE.findall(text):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.append(n)
    return out


def content_tokens(text: str) -> set[str]:
    return {
        w for w in (t.lower() for t in _WORD_RE.findall(strip_thinking(text)))
        if w not in _STOPWORDS and len(w) > 1
    }


def lexical_agreement(a: str, b: str) -> float:
    """Jaccard over content words; contradictory numbers force 0.0."""
    na, nb = set(extract_numbers(a)), set(extract_numbers(b))
    if na and nb and not (na & nb):
        return 0.0
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 1.0 if normalize_answer(a) == normalize_answer(b) else 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(slots=True)
class SummaryConstraint:
    kind: str  # "sentences" | "words" | "bullets"
    n: int
    exact: bool  # True -> exactly n, False -> at most n


_SENT_C_RE = re.compile(r"\bin\s+(?:exactly\s+)?(one|two|three|four|five|\d+)\s+sentences?\b", re.I)
_SINGLE_SENT_RE = re.compile(r"\b(?:in\s+)?(?:a|one)\s+(?:single\s+)?sentence\b", re.I)
_WORDS_MAX_RE = re.compile(
    r"\b(?:in\s+)?(?:under|at\s+most|no\s+more\s+than|maximum\s+(?:of\s+)?|fewer\s+than|less\s+than|within)\s+(\d+)\s+words?\b",
    re.I,
)
_WORDS_IN_RE = re.compile(r"\bin\s+(?:about\s+|around\s+|exactly\s+)?(\d+)\s+words?\b", re.I)
_BULLETS_RE = re.compile(r"\b(one|two|three|four|five|\d+)?\s*bullet\s*(?:points?|list)\b", re.I)


def _to_n(token: str | None, default: int = 0) -> int:
    if not token:
        return default
    token = token.lower()
    return _WORD_NUMS.get(token, 0) or (int(token) if token.isdigit() else default)


def parse_summary_constraint(prompt: str) -> SummaryConstraint | None:
    """Extract the length/format constraint a summarization prompt states, if any."""
    if m := _BULLETS_RE.search(prompt):
        n = _to_n(m.group(1), default=0)
        return SummaryConstraint("bullets", n, exact=bool(n))
    if m := _SENT_C_RE.search(prompt):
        return SummaryConstraint("sentences", _to_n(m.group(1), 1), exact=True)
    if _SINGLE_SENT_RE.search(prompt):
        return SummaryConstraint("sentences", 1, exact=True)
    if m := _WORDS_MAX_RE.search(prompt):
        return SummaryConstraint("words", int(m.group(1)), exact=False)
    if m := _WORDS_IN_RE.search(prompt):
        # "in N words" judged leniently; enforce as a ceiling with slack
        return SummaryConstraint("words", int(int(m.group(1)) * 1.15) + 2, exact=False)
    return None
