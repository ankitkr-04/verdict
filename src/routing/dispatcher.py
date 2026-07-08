"""Task -> category classification: scored heuristics, free local-LLM fallback on ambiguity."""

from __future__ import annotations

import re

from src.core.config import Config
from src.llm.local_llm import LocalError, LocalLLM
from src.core.schemas import Category

_KEYWORD_SCORE = 1
_PATTERN_SCORE = 2
_CLASSIFY_SNIPPET_CHARS = 500

_CLASSIFY_USER = (
    "Classify this task into exactly one category word from: {options}.\n"
    "- math: anything computable — arithmetic, dates, counting, unit conversion.\n"
    "- sentiment: ONLY when judging the emotional tone of a GIVEN piece of text.\n"
    "- summarize: ONLY when shortening or condensing GIVEN text.\n"
    "- ner: ONLY when extracting named entities from text.\n"
    "- factual: open questions, explanations, definitions, opinions, or comparisons "
    "(e.g. 'is X better than Y') — and anything that fits nothing else.\n"
    "Reply with one word only.\n\nTask: {snippet}"
)


class Dispatcher:
    def __init__(self, config: Config) -> None:
        self._min_score = config.dispatch_min_score
        self._llm_fallback = config.dispatch_llm_fallback
        self._priority = config.dispatch_priority
        self._keywords: dict[Category, list[re.Pattern[str]]] = {}
        self._patterns: dict[Category, list[re.Pattern[str]]] = {}
        for cat, pol in config.categories.items():
            self._keywords[cat] = [
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in pol.dispatch_keywords
            ]
            self._patterns[cat] = [
                re.compile(p, re.IGNORECASE) for p in pol.dispatch_patterns
            ]

    def score(self, prompt: str) -> dict[Category, int]:
        scores: dict[Category, int] = {}
        for cat in self._keywords:
            s = sum(_KEYWORD_SCORE for kw in self._keywords[cat] if kw.search(prompt))
            s += sum(_PATTERN_SCORE for p in self._patterns[cat] if p.search(prompt))
            scores[cat] = s
        return scores

    def _rank(self, cat: Category) -> int:
        try:
            return self._priority.index(cat)
        except ValueError:
            return len(self._priority)

    async def classify(self, prompt: str, local: LocalLLM) -> tuple[Category, str]:
        """Return (category, method) where method is 'heuristic', 'llm', or 'default'.

        Factual is the catch-all and never competes heuristically — its keywords are
        generic interrogatives that also fire inside every other category's prompts.
        Ties between specific categories break by configured priority, not an extra call.
        """
        scores = self.score(prompt)
        specific = {c: s for c, s in scores.items() if c is not Category.FACTUAL}
        best_score = max(specific.values(), default=0)
        if best_score >= self._min_score:
            leaders = [c for c, s in specific.items() if s == best_score]
            return min(leaders, key=self._rank), "heuristic"

        if self._llm_fallback:
            options = ", ".join(c.value for c in Category)
            user = _CLASSIFY_USER.format(options=options, snippet=prompt[:_CLASSIFY_SNIPPET_CHARS])
            try:
                resp = await local.generate(
                    "You label tasks with a single category word.", user,
                    temperature=0.0, max_tokens=8,
                )
                lowered = resp.text.lower()
                for cat in Category:
                    if cat.value in lowered:
                        return cat, "llm"
            except LocalError:
                pass

        # Unclassifiable -> factual: its solver is the most generic (answer + agreement gate).
        return Category.FACTUAL, "default"
