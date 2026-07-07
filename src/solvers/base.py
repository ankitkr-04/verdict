"""Solver framework: local generate -> free verify -> repair -> paid escalate.

BaseSolver.solve() owns the escalation decision and token accounting; subclasses only
implement attempt_local() — produce a locally-verified answer or explain why they can't.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from src.core.config import CategoryPolicy, Config
from src.core.schemas import LLMResponse, Route, SolveResult, Task
from src.llm.local_llm import LocalError, LocalLLM
from src.llm.remote_llm import RemoteError, RemoteLLM
from src.routing.budget import Budget, Mode
from src.routing.calibration import calibrated_confidence
from src.routing.prompt_compression import build_escalation_user

_FALLBACK_ANSWER = "Unable to determine a reliable answer."
_FEEDBACK_CAP = 600


@dataclass(slots=True)
class SolveContext:
    config: Config
    local: LocalLLM
    remote: RemoteLLM
    budget: Budget
    remaining_tasks: Callable[[], int]

    def mode(self) -> Mode:
        return self.budget.mode(self.remaining_tasks())


@dataclass(slots=True)
class LocalAttempt:
    answer: str | None = None   # best usable answer text (even if unverified)
    verified: bool = False
    repaired: bool = False
    confidence: float = 0.0
    feedback: str = ""          # why verification failed (ledger detail / escalation reason)
    local_calls: int = 0
    best_logprob: float | None = None


class BaseSolver(ABC):
    def __init__(self, policy: CategoryPolicy, ctx: SolveContext) -> None:
        self.policy = policy
        self.ctx = ctx

    # ---- helpers for subclasses -------------------------------------------------

    async def generate(
        self,
        user: str,
        attempt: LocalAttempt,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        p = self.policy
        suffix = "" if p.thinking else self.ctx.local.no_think_suffix
        resp = await self.ctx.local.generate(
            p.system if system is None else system,
            user + suffix,
            temperature=p.temperature if temperature is None else temperature,
            max_tokens=p.max_tokens if max_tokens is None else max_tokens,
            stop=stop,
        )
        attempt.local_calls += 1
        if resp.mean_logprob is not None:
            best = attempt.best_logprob
            attempt.best_logprob = resp.mean_logprob if best is None else max(best, resp.mean_logprob)
        return resp

    def render_user(self, task: Task) -> str:
        return self.policy.user_template.format(prompt=task.prompt)

    def render_repair(self, task: Task, previous: str, feedback: str) -> str:
        return self.policy.repair_template.format(
            prompt=task.prompt, previous=previous, feedback=feedback[:_FEEDBACK_CAP]
        )

    def format_answer(self, answer: str) -> str:
        return self.policy.answer_format.format(answer=answer.strip())

    def calibrated(self, mean_logprob: float | None) -> tuple[float, float]:
        """(calibrated confidence, theta) for this category."""
        a, b, theta = self.ctx.config.platt(self.policy.name)
        return calibrated_confidence(mean_logprob, a, b), theta

    def can_repair(self, mode: Mode, used: int) -> bool:
        return mode is Mode.FULL and used < self.policy.repair_attempts and bool(self.policy.repair_template)

    # ---- per-category logic -----------------------------------------------------

    @abstractmethod
    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt: ...

    def post_remote(self, task: Task, text: str) -> str:
        """Shape the remote answer for output; free local post-formatting can override."""
        return text.strip()

    # ---- shared solve flow ------------------------------------------------------

    async def solve(self, task: Task) -> SolveResult:
        t0 = time.monotonic()
        mode = self.ctx.mode()
        attempt = LocalAttempt()
        try:
            attempt = await self.attempt_local(task, mode)
        except LocalError as e:
            attempt.feedback = f"local backend error: {e}"

        def done(answer: str, route: Route, *, conf: float, rpt: int = 0, rct: int = 0,
                 rcalls: int = 0, detail: str = "") -> SolveResult:
            return SolveResult(
                task_id=task.task_id,
                answer=answer,
                category=self.policy.name,
                route=route,
                confidence=conf,
                remote_prompt_tokens=rpt,
                remote_completion_tokens=rct,
                local_calls=attempt.local_calls,
                remote_calls=rcalls,
                wall_ms=int((time.monotonic() - t0) * 1000),
                detail=detail,
                best_logprob=attempt.best_logprob,
            )

        if attempt.verified and attempt.answer:
            route = Route.LOCAL_REPAIR if attempt.repaired else Route.LOCAL
            return done(attempt.answer, route, conf=attempt.confidence)

        # Verified local failure (or nothing verifiable) -> the only paid path.
        user = build_escalation_user(task.prompt, self.policy.escalate.instruction)
        try:
            resp = await self.ctx.remote.complete(
                user,
                max_tokens=self.policy.escalate.max_tokens,
                temperature=self.policy.escalate.temperature,
            )
            answer = self.post_remote(task, resp.text)
            if answer:
                return done(
                    answer, Route.ESCALATED, conf=0.9, rcalls=1,
                    rpt=resp.prompt_tokens, rct=resp.completion_tokens,
                    detail=attempt.feedback,
                )
            # Remote replied empty: count the call, fall through to local fallback.
            return done(
                attempt.answer or _FALLBACK_ANSWER, Route.REMOTE_FAILED,
                conf=attempt.confidence, rcalls=1,
                rpt=resp.prompt_tokens, rct=resp.completion_tokens,
                detail="remote returned empty text",
            )
        except RemoteError as e:
            return done(
                attempt.answer or _FALLBACK_ANSWER, Route.REMOTE_FAILED,
                conf=attempt.confidence, detail=f"{attempt.feedback} | {e}",
            )
