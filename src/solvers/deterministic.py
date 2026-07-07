"""Zero-LLM answer lane: strict exact-match handlers computed in pure Python.

This is a bonus fast path, not the routing strategy — the general lane for computable
tasks is the PAL math solver (LLM translates to code, executor computes). A handler
here fires only when the whole prompt matches a narrow pattern it can answer with
certainty; anything ambiguous returns None and flows to the normal solvers.

Cost: zero tokens (local and remote), microseconds of wall-clock, no hallucination.
"""

from __future__ import annotations

import ast
import calendar
import operator
import re
from datetime import date, datetime, timedelta

_MAX_ABS_RESULT = 1e15
_MAX_EXPONENT = 12

# ---------------------------------------------------------------- formatting

def _fmt_num(x: float | int) -> str:
    if isinstance(x, float):
        if x != x or abs(x) == float("inf"):  # nan/inf: not an answer
            raise ValueError("non-finite result")
        if x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return str(x)


# ---------------------------------------------------------------- safe arithmetic

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node: ast.expr) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        result = _BIN_OPS[type(node.op)](left, right)
        if abs(result) > _MAX_ABS_RESULT:
            raise ValueError("result too large")
        return result
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def _safe_eval(expr: str) -> float | int:
    return _eval_node(ast.parse(expr, mode="eval").body)


_ARITH_RE = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|compute|evaluate|solve)[:\s]+"
    r"([\d\s+\-*/().^]+?)\s*[?.!=]*\s*$",
    re.IGNORECASE,
)


def _arithmetic(prompt: str) -> str | None:
    m = _ARITH_RE.match(prompt)
    if not m:
        return None
    expr = m.group(1).replace("^", "**")
    # Require an actual computation: at least two numbers joined by an operator.
    if len(re.findall(r"\d+(?:\.\d+)?", expr)) < 2:
        return None
    try:
        return _fmt_num(_safe_eval(expr))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


_PERCENT_RE = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|compute|find)\s+"
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:[,\d]*\d)?(?:\.\d+)?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _percent_of(prompt: str) -> str | None:
    m = _PERCENT_RE.match(prompt)
    if not m:
        return None
    pct = float(m.group(1))
    base = float(m.group(2).replace(",", ""))
    try:
        return _fmt_num(pct * base / 100)
    except ValueError:
        return None


# ---------------------------------------------------------------- dates

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})

_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_MDY_RE = re.compile(r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_DATE_DMY_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]+),?\s+(\d{4})\b")


def _parse_dates(text: str) -> list[date]:
    """Every unambiguous date in the text, in order of appearance."""
    found: list[tuple[int, date]] = []
    for m in _DATE_ISO_RE.finditer(text):
        try:
            found.append((m.start(), date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
        except ValueError:
            pass
    for m in _DATE_MDY_RE.finditer(text):
        month = _MONTHS.get(m.group(1).lower())
        if month:
            try:
                found.append((m.start(), date(int(m.group(3)), month, int(m.group(2)))))
            except ValueError:
                pass
    for m in _DATE_DMY_RE.finditer(text):
        month = _MONTHS.get(m.group(2).lower())
        if month:
            try:
                found.append((m.start(), date(int(m.group(3)), month, int(m.group(1)))))
            except ValueError:
                pass
    found.sort(key=lambda t: t[0])
    dates: list[date] = []
    for _, d in found:
        if d not in dates:
            dates.append(d)
    return dates


_DAY_OF_WEEK_RE = re.compile(
    r"^\s*(?:what|which)\s+day\s+of\s+the\s+week\s+(?:is|was|will\s+.*\s+be|does\s+.*\s+fall\s+on)\b",
    re.IGNORECASE,
)


def _day_of_week(prompt: str) -> str | None:
    if not _DAY_OF_WEEK_RE.match(prompt):
        return None
    dates = _parse_dates(prompt)
    if len(dates) != 1:
        return None
    return dates[0].strftime("%A")


_DAYS_BETWEEN_RE = re.compile(
    r"^\s*how\s+many\s+days\s+(?:are\s+(?:there\s+)?)?between\b", re.IGNORECASE
)


def _days_between(prompt: str) -> str | None:
    if not _DAYS_BETWEEN_RE.match(prompt):
        return None
    dates = _parse_dates(prompt)
    if len(dates) != 2:
        return None
    return str(abs((dates[1] - dates[0]).days))


_DATE_OFFSET_RE = re.compile(
    r"^\s*(?:what\s+(?:is\s+the\s+)?date\s+(?:is|was|will\s+it\s+be|falls?)\s+)?"
    r"(\d+)\s+days?\s+(after|before|from)\b",
    re.IGNORECASE,
)


def _date_offset(prompt: str) -> str | None:
    m = _DATE_OFFSET_RE.match(prompt)
    if not m:
        return None
    dates = _parse_dates(prompt)
    if len(dates) != 1:
        return None
    delta = timedelta(days=int(m.group(1)))
    result = dates[0] - delta if m.group(2).lower() == "before" else dates[0] + delta
    return result.isoformat()


# ---------------------------------------------------------------- strings & counting

_QUOTED = r"['\"‘’“”]"

_COUNT_LETTER_RE = re.compile(
    rf"^\s*how\s+many\s+(?:times\s+does\s+the\s+letter\s+{_QUOTED}?(\w){_QUOTED}?\s+"
    rf"(?:appear|occur)s?\s+in|{_QUOTED}?(\w){_QUOTED}?\s*'?s\s+are\s+(?:there\s+)?in)\s+"
    rf"(?:the\s+word\s+)?{_QUOTED}?([\w\s-]+?){_QUOTED}?\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _count_letter(prompt: str) -> str | None:
    m = _COUNT_LETTER_RE.match(prompt)
    if not m:
        return None
    letter = (m.group(1) or m.group(2)).lower()
    text = m.group(3)
    return str(text.lower().count(letter))


_COUNT_KIND_RE = re.compile(
    rf"^\s*how\s+many\s+(vowels|consonants|letters|characters|words)\s+"
    rf"(?:are\s+(?:there\s+)?in|does)\s+(?:the\s+(?:word|string|phrase|sentence)\s+)?"
    rf"{_QUOTED}(.+?){_QUOTED}\s*(?:have|contain)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _count_kind(prompt: str) -> str | None:
    m = _COUNT_KIND_RE.match(prompt)
    if not m:
        return None
    kind, text = m.group(1).lower(), m.group(2)
    if kind == "vowels":
        return str(sum(c in "aeiou" for c in text.lower()))
    if kind == "consonants":
        return str(sum(c.isalpha() and c not in "aeiou" for c in text.lower()))
    if kind == "letters":
        return str(sum(c.isalpha() for c in text))
    if kind == "characters":
        return str(len(text))
    return str(len(text.split()))


_REVERSE_RE = re.compile(
    rf"^\s*(?:reverse\s+the\s+(?:string|word|letters\s+(?:of|in))\s+"
    rf"{_QUOTED}?([\w-]+){_QUOTED}?|(?:what\s+is\s+)?{_QUOTED}?([\w-]+){_QUOTED}?\s+"
    rf"spell(?:ed|t)?\s+backwards?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _reverse(prompt: str) -> str | None:
    m = _REVERSE_RE.match(prompt)
    if not m:
        return None
    return (m.group(1) or m.group(2))[::-1]


# ---------------------------------------------------------------- entry point

_HANDLERS = (
    _arithmetic,
    _percent_of,
    _day_of_week,
    _days_between,
    _date_offset,
    _count_letter,
    _count_kind,
    _reverse,
)


def try_deterministic(prompt: str) -> str | None:
    """The answer if a strict handler fully matches the prompt, else None.

    Handlers are anchored to the whole prompt on purpose: a partial match on a word
    problem must never fire. False negatives are free (normal solvers take over);
    a false positive would be a wrong final answer.
    """
    if len(prompt) > 400:  # long prompts are never the trivial phrasings we match
        return None
    text = " ".join(prompt.split())
    for handler in _HANDLERS:
        try:
            answer = handler(text)
        except Exception:  # a handler bug must never take down the pipeline
            answer = None
        if answer is not None:
            return answer
    return None
