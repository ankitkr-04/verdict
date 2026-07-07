"""Solvers gated by agreement + calibrated confidence: logic puzzles and factual QA.

Factual is the designed main escalation source — no deterministic oracle exists, so the
calibrated logprob gate (theta from config/calibration.json) decides local vs remote.
"""

from __future__ import annotations

import asyncio

from src.core.parsing import extract_final, lexical_agreement, strip_thinking
from src.core.schemas import LLMResponse, Task
from src.routing.budget import Mode
from src.solvers.base import BaseSolver, LocalAttempt
from src.verification.checks import majority

_CONF_SINGLE_FLOOR = 0.0


class LogicSolver(BaseSolver):
    """Self-consistency: N diverse samples, majority vote on the FINAL line."""

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        n = self.policy.n_samples if mode is Mode.FULL else 1
        user = self.render_user(task)

        if n > 1:
            responses = await asyncio.gather(*(self.generate(user, a) for _ in range(n)))
            finals = [f for r in responses if (f := extract_final(strip_thinking(r.text)))]
            if not finals:
                a.feedback = "no sample produced a FINAL line"
                return a
            winner, votes = majority(finals)
            a.answer = self.format_answer(winner)
            needed = int(self.policy.extra_num("majority_min", 2))
            a.verified = votes >= needed
            a.confidence = votes / n
            if not a.verified:
                a.feedback = f"majority {votes}/{n} below required {needed}"
            return a

        resp = await self.generate(user, a, temperature=0.2)
        final = extract_final(strip_thinking(resp.text))
        if not final:
            a.feedback = "no FINAL line"
            return a
        conf, theta = self.calibrated(a.best_logprob)
        a.answer = self.format_answer(final)
        a.verified = conf >= theta
        a.confidence = conf
        if not a.verified:
            a.feedback = f"single-sample confidence {conf:.2f} < theta {theta:.2f}"
        return a


class FactualSolver(BaseSolver):
    """Two samples must agree lexically AND clear the Platt-calibrated logprob gate."""

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        n = self.policy.n_samples if mode is Mode.FULL else 1
        user = self.render_user(task)
        responses = await asyncio.gather(*(self.generate(user, a) for _ in range(n)))
        texts = [strip_thinking(r.text).strip() for r in responses]
        usable = [(t, r) for t, r in zip(texts, responses) if t]
        if not usable:
            a.feedback = "empty local answers"
            return a

        primary_text, _ = max(usable, key=lambda tr: self._lp(tr[1]))
        conf, theta = self.calibrated(a.best_logprob)
        a.answer = self.format_answer(primary_text)
        a.confidence = conf

        if len(usable) >= 2:
            agree = lexical_agreement(usable[0][0], usable[1][0])
            agree_min = self.policy.extra_num("agreement_min", 0.45)
            a.verified = agree >= agree_min and conf >= theta
            a.confidence = (conf + agree) / 2
            if not a.verified:
                a.feedback = (
                    f"agreement {agree:.2f} (min {agree_min}) / confidence {conf:.2f} (theta {theta:.2f})"
                )
        else:
            a.verified = conf >= theta
            if not a.verified:
                a.feedback = f"single-sample confidence {conf:.2f} < theta {theta:.2f}"
        return a

    @staticmethod
    def _lp(resp: LLMResponse) -> float:
        return resp.mean_logprob if resp.mean_logprob is not None else float("-inf")
