"""Task -> category classification.

LLM-first: local tokens are free (only remote is scored) and regex pattern-matching
is brittle on nuance/paraphrase, so we let the local model read the task and name the
category. Trivial computables are answered even earlier by the deterministic lane
(run before classification in the orchestrator), so they never spend a classify call.
"""

from __future__ import annotations

from src.core.schemas import Category
from src.llm.local_llm import LocalError, LocalLLM

_CLASSIFY_SNIPPET_CHARS = 500

_CLASSIFY_SYSTEM = "You label a task with a single category word."

_CLASSIFY_USER = (
    "Classify this task into exactly one category word from: {options}.\n"
    "- math: anything computable — arithmetic, word problems, dates, counting, unit conversion.\n"
    "- code_gen: writing new code from a description.\n"
    "- code_debug: fixing or finding a bug in code that is given.\n"
    "- ner: extracting named entities (people, orgs, locations, dates) from text.\n"
    "- sentiment: judging the emotional tone of a GIVEN piece of text.\n"
    "- summarize: shortening or condensing GIVEN text.\n"
    "- logic: puzzles, deduction, or multi-step reasoning with one definite answer.\n"
    "- factual: open questions, explanations, definitions, opinions, or comparisons "
    "(e.g. 'is X better than Y') — and anything that fits nothing else.\n"
    "Reply with one word only.\n\nTask: {snippet}"
)

# Longest values first so 'code_debug' / 'code_gen' match before any shorter overlap.
_ORDERED = sorted(Category, key=lambda c: -len(c.value))


class Dispatcher:
    """Stateless LLM classifier (kept as a class for a stable call site)."""

    async def classify(self, prompt: str, local: LocalLLM) -> tuple[Category, str]:
        """Return (category, method) where method is 'llm' or 'default'."""
        options = ", ".join(c.value for c in Category)
        user = _CLASSIFY_USER.format(options=options, snippet=prompt[:_CLASSIFY_SNIPPET_CHARS])
        try:
            resp = await local.generate(
                _CLASSIFY_SYSTEM, user, temperature=0.0, max_tokens=8,
            )
            lowered = resp.text.lower()
            for cat in _ORDERED:
                if cat.value in lowered:
                    return cat, "llm"
        except LocalError:
            pass
        # Unclassifiable (or local error) -> factual: the most generic solver.
        return Category.FACTUAL, "default"
