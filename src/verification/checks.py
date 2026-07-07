"""Free verification primitives: deterministic exec, schema, constraint, and agreement checks.

These are the reason local answers can be trusted without spending remote tokens.
Solvers compose them; nothing here calls a model.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import resource
import sys
from collections import Counter
from dataclasses import dataclass

from jsonschema import Draft202012Validator

from src.core import settings
from src.core.parsing import SummaryConstraint, count_bullets, count_sentences, count_words, normalize_answer

# ---------------------------------------------------------------- python exec

@dataclass(slots=True)
class ExecResult:
    ok: bool
    stdout: str
    stderr: str


def _child_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


async def run_python(code: str, timeout_s: float = settings.PYEXEC_TIMEOUT_S) -> ExecResult:
    """Run code in an isolated interpreter (-I) with CPU/memory rlimits and a hard timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_child_limits,
        )
    except OSError as e:
        return ExecResult(False, "", f"spawn failed: {e!r}")
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return ExecResult(False, "", f"timed out after {timeout_s}s")
    cap = settings.PYEXEC_MAX_OUTPUT_CHARS
    return ExecResult(
        ok=proc.returncode == 0,
        stdout=out.decode(errors="replace")[:cap].strip(),
        stderr=err.decode(errors="replace")[-cap:].strip(),
    )


def ensure_prints(code: str) -> str:
    """If the snippet ends in a bare expression and never prints, print that expression."""
    if "print(" in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    expr_src = ast.unparse(last.value)
    kept = code.splitlines()[: last.lineno - 1]
    return "\n".join([*kept, f"print({expr_src})"])


async def exec_math(code: str) -> ExecResult:
    return await run_python(ensure_prints(code))


def code_parses(code: str) -> str | None:
    """None if the code parses, else the SyntaxError message (repair feedback)."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


async def run_asserts(func_code: str, asserts_code: str) -> ExecResult:
    sentinel = "VERDICT_ASSERTS_OK"
    program = f"{func_code}\n\n{asserts_code}\n\nprint({sentinel!r})"
    result = await run_python(program)
    if result.ok and sentinel not in result.stdout:
        return ExecResult(False, result.stdout, "asserts did not complete")
    return result


_EXPECTED_OUT_RE = re.compile(
    r"expected output\s*:?\s*\n?(.+?)(?:\n\s*\n|$)", re.IGNORECASE | re.DOTALL
)


def extract_expected_output(prompt: str) -> str | None:
    m = _EXPECTED_OUT_RE.search(prompt)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- NER

def entity_schema(types: list[str]) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["text", "type"],
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": types},
            },
        },
    }


def validate_entities(value: object, types: list[str]) -> str | None:
    """None if valid, else a short feedback string for the repair prompt."""
    if not isinstance(value, list):
        return "output is not a JSON array"
    errors = sorted(
        Draft202012Validator(entity_schema(types)).iter_errors(value),
        key=lambda e: str(e.path),
    )
    if errors:
        e = errors[0]
        return f"schema violation at {list(e.path)}: {e.message}"
    return None


def entity_key_set(entities: list) -> set[tuple[str, str]]:
    return {
        (str(e.get("text", "")).strip().lower(), str(e.get("type", "")).strip().upper())
        for e in entities
        if isinstance(e, dict)
    }


def entity_f1(a: list, b: list) -> float:
    ka, kb = entity_key_set(a), entity_key_set(b)
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    overlap = len(ka & kb)
    return 2 * overlap / (len(ka) + len(kb))


# ---------------------------------------------------------------- summaries

def check_summary(text: str, c: SummaryConstraint | None) -> str | None:
    """None if the summary satisfies the constraint, else feedback for repair."""
    if not text.strip():
        return "summary is empty"
    if c is None:
        return None
    if c.kind == "sentences":
        got = count_sentences(text)
        if got != c.n:
            return f"required exactly {c.n} sentence(s), got {got}"
    elif c.kind == "words":
        got = count_words(text)
        if got > c.n:
            return f"required at most {c.n} words, got {got}"
    elif c.kind == "bullets":
        got = count_bullets(text)
        if c.exact and got != c.n:
            return f"required exactly {c.n} bullet points, got {got}"
        if not c.exact and got < 2:
            return f"required bullet points, got {got}"
    return None


# ---------------------------------------------------------------- agreement

def majority(answers: list[str]) -> tuple[str, int]:
    """(representative original answer, votes) for the most common normalized answer."""
    keyed = [(normalize_answer(a), a) for a in answers if a and a.strip()]
    if not keyed:
        return "", 0
    counts = Counter(k for k, _ in keyed)
    winner_key, votes = counts.most_common(1)[0]
    representative = next(orig for k, orig in keyed if k == winner_key)
    return representative, votes
