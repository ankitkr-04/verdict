"""Shared dataclasses and enums passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZE = "summarize"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    LOGIC = "logic"
    CODE_GEN = "code_gen"


class Route(StrEnum):
    """How a task's final answer was produced."""

    LOCAL = "local"                # local answer passed verification
    LOCAL_REPAIR = "local_repair"  # passed after a repair round
    LOCAL_UNVERIFIED = "local_unverified"  # accepted without full verification (panic/degraded)
    ESCALATED = "escalated"        # remote answer used
    REMOTE_FAILED = "remote_failed"  # escalation attempted, remote errored -> best local kept
    INSURANCE = "insurance"        # end-of-run guaranteed-Fireworks-call task
    ERROR = "error"                # solver crashed -> fallback answer


@dataclass(slots=True)
class Task:
    task_id: str
    prompt: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mean_logprob: float | None = None
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class VerifyOutcome:
    passed: bool
    confidence: float = 0.0
    answer: str | None = None  # verifier-derived final answer (e.g. exec stdout)
    feedback: str = ""  # machine feedback fed into repair prompts


@dataclass(slots=True)
class SolveResult:
    task_id: str
    answer: str
    category: Category
    route: Route
    confidence: float = 0.0
    remote_prompt_tokens: int = 0
    remote_completion_tokens: int = 0
    local_calls: int = 0
    remote_calls: int = 0
    wall_ms: int = 0
    detail: str = ""
    best_logprob: float | None = None  # raw signal for offline Platt calibration
