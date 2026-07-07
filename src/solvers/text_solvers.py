"""Solvers verified by structure and agreement: NER, summarization, sentiment."""

from __future__ import annotations

import asyncio
import json

from src.core.parsing import extract_json_array, extract_label, parse_summary_constraint, strip_thinking
from src.core.schemas import Task
from src.routing.budget import Mode
from src.solvers.base import BaseSolver, LocalAttempt
from src.verification.checks import check_summary, entity_f1, validate_entities

_CONF_AGREE = 0.9           # independent samples agreed
_CONF_SINGLE_VALID = 0.7    # structurally valid but only one sample (degraded mode)
_CONF_MAJORITY_2_OF_3 = 0.75
_CONF_CONSTRAINED = 0.85    # summary satisfies its deterministic constraint
_CONF_UNCONSTRAINED = 0.7   # nothing checkable; escalation would buy nothing verifiable


class NERSolver(BaseSolver):
    """JSON-schema validation + cross-sample entity F1; canonical JSON out."""

    def _types(self) -> list[str]:
        return [str(t) for t in self.policy.extra.get("entity_types", [])]

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        n = self.policy.n_samples if mode is Mode.FULL else 1
        user = self.render_user(task)
        responses = await asyncio.gather(*(self.generate(user, a) for _ in range(n)))

        parsed: list[list] = []
        first_error = ""
        for resp in responses:
            value = extract_json_array(strip_thinking(resp.text))
            err = validate_entities(value, self._types())
            if err is None and value is not None:
                parsed.append(value)
            elif not first_error:
                first_error = err or "no JSON array found"

        if not parsed and self.can_repair(mode, 0):
            previous = strip_thinking(responses[0].text)[:500]
            resp = await self.generate(self.render_repair(task, previous, first_error), a)
            value = extract_json_array(strip_thinking(resp.text))
            if validate_entities(value, self._types()) is None and value is not None:
                parsed.append(value)
                a.repaired = True

        if not parsed:
            a.feedback = first_error or "no valid entity JSON"
            return a

        primary = parsed[0]
        if len(parsed) >= 2:
            f1 = entity_f1(parsed[0], parsed[1])
            threshold = self.policy.extra_num("agreement_f1", 0.8)
            if f1 < threshold:
                a.feedback = f"entity agreement F1 {f1:.2f} below {threshold}"
                return a
            a.confidence = max(_CONF_AGREE, f1)
        else:
            a.confidence = _CONF_SINGLE_VALID

        a.answer = self.format_answer(json.dumps(primary, ensure_ascii=False))
        a.verified = True
        return a

    def post_remote(self, task: Task, text: str) -> str:
        value = extract_json_array(text)
        if value is not None and validate_entities(value, self._types()) is None:
            return json.dumps(value, ensure_ascii=False)
        return text.strip()


class SummarizeSolver(BaseSolver):
    """Deterministic length/format constraint check with a repair loop until compliant."""

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        constraint = parse_summary_constraint(task.prompt)
        resp = await self.generate(self.render_user(task), a)
        text = strip_thinking(resp.text)

        rounds = 0
        feedback = check_summary(text, constraint)
        while feedback is not None and self.can_repair(mode, rounds):
            resp = await self.generate(self.render_repair(task, text, feedback), a)
            text = strip_thinking(resp.text)
            a.repaired = True
            rounds += 1
            feedback = check_summary(text, constraint)

        if feedback is None:
            a.answer = self.format_answer(text)
            a.verified = True
            a.confidence = _CONF_CONSTRAINED if constraint else _CONF_UNCONSTRAINED
        else:
            a.answer = self.format_answer(text)  # best local candidate for fallback
            a.feedback = feedback
        return a


class SentimentSolver(BaseSolver):
    """Label + justification; second-sample label agreement, third-sample tie-break."""

    def _labels(self) -> list[str]:
        return [str(l) for l in self.policy.extra.get("labels", [])]

    def _extract(self, text: str) -> str | None:
        return extract_label(text, self._labels())

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        n = self.policy.n_samples if mode is Mode.FULL else 1
        user = self.render_user(task)
        responses = await asyncio.gather(*(self.generate(user, a) for _ in range(n)))
        texts = [strip_thinking(r.text) for r in responses]
        labels = [self._extract(t) for t in texts]

        if labels[0] is None and self.can_repair(mode, 0):
            resp = await self.generate(
                self.render_repair(task, texts[0][:300], "no valid label on line 1"), a
            )
            texts[0] = strip_thinking(resp.text)
            labels[0] = self._extract(texts[0])
            a.repaired = True

        valid = [(lbl, txt) for lbl, txt in zip(labels, texts) if lbl is not None]
        if not valid:
            a.feedback = "no parsable sentiment label"
            return a

        if n == 1:
            conf, theta = self.calibrated(a.best_logprob)
            a.answer = self.format_answer(valid[0][1])
            a.verified = conf >= theta
            a.confidence = conf
            if not a.verified:
                a.feedback = f"single-sample confidence {conf:.2f} < theta {theta:.2f}"
            return a

        if len(valid) >= 2 and valid[0][0] == valid[1][0]:
            a.answer = self.format_answer(valid[0][1])
            a.verified = True
            a.confidence = _CONF_AGREE
            return a

        # Disagreement -> one tie-break sample, majority of three wins.
        if mode is Mode.FULL:
            resp = await self.generate(user, a)
            text3 = strip_thinking(resp.text)
            label3 = self._extract(text3)
            votes = [lbl for lbl, _ in valid] + ([label3] if label3 else [])
            for lbl in set(votes):
                if votes.count(lbl) >= 2:
                    winner_text = next(t for l, t in [*valid, (label3, text3)] if l == lbl)
                    a.answer = self.format_answer(winner_text)
                    a.verified = True
                    a.confidence = _CONF_MAJORITY_2_OF_3
                    return a
        a.answer = self.format_answer(valid[0][1])
        a.feedback = "sentiment samples disagreed"
        return a
