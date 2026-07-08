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
import math
import operator
import re
from datetime import date, timedelta

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


# ---------------------------------------------------------------- number lists

_NUM = r"-?\d+(?:\.\d+)?"
_LIST_STATS_RE = re.compile(
    rf"^\s*(?:what\s+is|what's|calculate|compute|find)\s+the\s+"
    rf"(average|mean|median|sum|total|minimum|maximum|smallest|largest|range)\s+"
    rf"of(?:\s+the\s+numbers?)?[:\s]+({_NUM}(?:\s*(?:,|and)\s+{_NUM})+)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _list_stats(prompt: str) -> str | None:
    m = _LIST_STATS_RE.match(prompt)
    if not m:
        return None
    op = m.group(1).lower()
    nums = [float(x) for x in re.findall(_NUM, m.group(2))]
    if len(nums) < 2:
        return None
    if op in ("average", "mean"):
        value = sum(nums) / len(nums)
    elif op == "median":
        s = sorted(nums)
        mid = len(s) // 2
        value = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    elif op in ("sum", "total"):
        value = sum(nums)
    elif op in ("minimum", "smallest"):
        value = min(nums)
    elif op in ("maximum", "largest"):
        value = max(nums)
    else:  # range
        value = max(nums) - min(nums)
    return _fmt_num(value)


_GCD_LCM_RE = re.compile(
    rf"^\s*(?:what\s+is|what's|calculate|compute|find)\s+the\s+"
    rf"(gcd|hcf|lcm|greatest\s+common\s+(?:divisor|factor)|highest\s+common\s+factor|"
    rf"least\s+common\s+multiple|lowest\s+common\s+multiple)\s+"
    rf"of\s+(\d+)\s+(?:,|and)\s+(\d+)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _gcd_lcm(prompt: str) -> str | None:
    m = _GCD_LCM_RE.match(prompt)
    if not m:
        return None
    op = m.group(1).lower()
    a, b = int(m.group(2)), int(m.group(3))
    if a == 0 or b == 0 or a > 10**9 or b > 10**9:
        return None
    if "lcm" in op or "multiple" in op:
        return str(a * b // math.gcd(a, b))
    return str(math.gcd(a, b))


_FACTORIAL_RE = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|compute)\s+"
    r"(?:the\s+factorial\s+of\s+(\d+)|(\d+)\s*(?:factorial|!))\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _factorial(prompt: str) -> str | None:
    m = _FACTORIAL_RE.match(prompt)
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    if n > 20:  # beyond this the expected format (sci notation?) is ambiguous
        return None
    return str(math.factorial(n))


_ROOT_RE = re.compile(
    rf"^\s*(?:what\s+is|what's|calculate|compute|find)\s+the\s+"
    rf"(square|cube)\s+root\s+of\s+({_NUM})\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _root(prompt: str) -> str | None:
    m = _ROOT_RE.match(prompt)
    if not m:
        return None
    x = float(m.group(2))
    if x < 0:
        return None
    value = x ** (1 / 2 if m.group(1).lower() == "square" else 1 / 3)
    rounded = round(value)
    if abs(rounded**2 - x) < 1e-9 if m.group(1).lower() == "square" else abs(rounded**3 - x) < 1e-9:
        return str(rounded)  # exact roots stay integers
    return _fmt_num(round(value, 4))


_PCT_CHANGE_RE = re.compile(
    rf"^\s*(?:what\s+is|what's|calculate|compute|find)\s+the\s+"
    rf"percent(?:age)?\s+(increase|decrease|change)\s+from\s+({_NUM})\s+to\s+({_NUM})\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _percent_change(prompt: str) -> str | None:
    m = _PCT_CHANGE_RE.match(prompt)
    if not m:
        return None
    old, new = float(m.group(2)), float(m.group(3))
    if old == 0:
        return None
    change = (new - old) / old * 100
    if m.group(1).lower() == "decrease":
        change = -change
    return _fmt_num(round(change, 4)) + "%"


# ---------------------------------------------------------------- unit conversion

# unit -> (canonical per-unit factor, dimension). Linear units only; temperature
# is handled separately. Aliases share a dimension so cross-dimension never fires.
_UNITS: dict[str, tuple[float, str]] = {
    "kilometer": (1000.0, "len"), "km": (1000.0, "len"),
    "meter": (1.0, "len"), "metre": (1.0, "len"), "m": (1.0, "len"),
    "centimeter": (0.01, "len"), "cm": (0.01, "len"),
    "millimeter": (0.001, "len"), "mm": (0.001, "len"),
    "mile": (1609.344, "len"), "mi": (1609.344, "len"),
    "yard": (0.9144, "len"), "foot": (0.3048, "len"), "feet": (0.3048, "len"),
    "ft": (0.3048, "len"), "inch": (0.0254, "len"), "inches": (0.0254, "len"),
    "kilogram": (1000.0, "mass"), "kg": (1000.0, "mass"),
    "gram": (1.0, "mass"), "g": (1.0, "mass"),
    "pound": (453.59237, "mass"), "lb": (453.59237, "mass"), "lbs": (453.59237, "mass"),
    "ounce": (28.349523125, "mass"), "oz": (28.349523125, "mass"),
    "liter": (1.0, "vol"), "litre": (1.0, "vol"), "l": (1.0, "vol"),
    "milliliter": (0.001, "vol"), "ml": (0.001, "vol"),
    "gallon": (3.785411784, "vol"),
    "hour": (3600.0, "time"), "minute": (60.0, "time"), "second": (1.0, "time"),
    "day": (86400.0, "time"), "week": (604800.0, "time"),
}

_CONVERT_RE = re.compile(
    rf"^\s*(?:convert\s+|what\s+is\s+|what's\s+|how\s+many\s+\w+\s+(?:is|are|in)\s+)?"
    rf"({_NUM})\s*([a-zA-Z]+)\s+(?:to|in|into)\s+([a-zA-Z]+)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_TEMP_RE = re.compile(
    rf"^\s*(?:convert\s+|what\s+is\s+|what's\s+)?({_NUM})\s*(?:°|degrees?\s*)?"
    rf"(celsius|fahrenheit|kelvin|c|f|k)\s+(?:to|in|into)\s+(?:°|degrees?\s*)?"
    rf"(celsius|fahrenheit|kelvin|c|f|k)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _unit_key(word: str) -> str:
    w = word.lower()
    return w[:-1] if w.endswith("s") and w[:-1] in _UNITS else w


def _convert(prompt: str) -> str | None:
    m = _TEMP_RE.match(prompt)
    if m:
        x = float(m.group(1))
        src, dst = m.group(2).lower()[0], m.group(3).lower()[0]
        if src == dst:
            return _fmt_num(x)
        as_c = {"c": x, "f": (x - 32) * 5 / 9, "k": x - 273.15}[src]
        out = {"c": as_c, "f": as_c * 9 / 5 + 32, "k": as_c + 273.15}[dst]
        return _fmt_num(round(out, 2))
    m = _CONVERT_RE.match(prompt)
    if not m:
        return None
    src, dst = _unit_key(m.group(2)), _unit_key(m.group(3))
    if src not in _UNITS or dst not in _UNITS or src == dst:
        return None
    (f_src, dim_src), (f_dst, dim_dst) = _UNITS[src], _UNITS[dst]
    if dim_src != dim_dst:
        return None  # cross-dimension: defer, never guess
    value = float(m.group(1)) * f_src / f_dst
    return _fmt_num(round(value, 4))


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

_ALPHA_SORT_RE = re.compile(
    r"^\s*(?:arrange|sort|put|list)\s+(?:the\s+(?:following\s+)?words?\s+)?"
    r"(?:in(?:to)?\s+)?alphabetical\s+order\s*[:\s]+([\w\s,'-]+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def _alpha_sort(prompt: str) -> str | None:
    m = _ALPHA_SORT_RE.match(prompt)
    if not m:
        return None
    words = [w.strip().strip("'\"") for w in re.split(r",|\band\b", m.group(1)) if w.strip()]
    if len(words) < 2 or any(" " in w for w in words):
        return None
    return ", ".join(sorted(words, key=str.lower))


_HANDLERS = (
    _arithmetic,
    _percent_of,
    _percent_change,
    _list_stats,
    _gcd_lcm,
    _factorial,
    _root,
    _convert,
    _day_of_week,
    _days_between,
    _date_offset,
    _count_letter,
    _count_kind,
    _reverse,
    _alpha_sort,
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
