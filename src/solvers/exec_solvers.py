"""Solvers verified by actually executing code: math (PAL), code generation, code debugging."""

from __future__ import annotations

from src.core.parsing import extract_code_block, extract_code_or_all, strip_thinking
from src.core.schemas import Task
from src.routing.budget import Mode
from src.solvers.base import BaseSolver, LocalAttempt
from src.verification.checks import (
    code_parses,
    exec_math,
    extract_expected_output,
    run_asserts,
    run_python,
)

_CONF_EXEC_OK = 0.95        # deterministic execution produced the answer
_CONF_TESTS_PASS = 0.9      # generated code passed extracted asserts
_CONF_RUNS_CLEAN = 0.75     # code executes without error (no oracle to compare against)
_CONF_PARSE_ONLY = 0.65     # syntactically valid, nothing else checkable
_MAX_ANSWER_CHARS = 200     # a math answer longer than this is garbage, not an answer
_NO_TESTS_SENTINEL = "pass"


class MathSolver(BaseSolver):
    """LLM writes a tiny program; Python computes the answer — verification is free and exact."""

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        resp = await self.generate(self.render_user(task), a)
        code = extract_code_or_all(strip_thinking(resp.text))
        result = await exec_math(code)

        if (not result.ok or not result.stdout) and self.can_repair(mode, 0):
            feedback = result.stderr or "the code produced no output"
            resp = await self.generate(self.render_repair(task, code, feedback), a)
            code = extract_code_or_all(strip_thinking(resp.text))
            result = await exec_math(code)
            a.repaired = True

        if result.ok and result.stdout and len(result.stdout) <= _MAX_ANSWER_CHARS:
            a.answer = self.format_answer(result.stdout)
            a.verified = True
            a.confidence = _CONF_EXEC_OK
        else:
            a.feedback = (result.stderr or "no usable output from computed code")[:300]
        return a


class CodeGenSolver(BaseSolver):
    """Generate a function, self-extract assert tests from the spec, run them."""

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        resp = await self.generate(self.render_user(task), a)
        code = extract_code_or_all(strip_thinking(resp.text))

        syntax_err = code_parses(code)
        if syntax_err and self.can_repair(mode, 0):
            resp = await self.generate(self.render_repair(task, code, syntax_err), a)
            code = extract_code_or_all(strip_thinking(resp.text))
            syntax_err = code_parses(code)
            a.repaired = True
        if syntax_err:
            a.feedback = syntax_err
            return a

        a.answer = self.format_answer(code)
        a.verified = True
        a.confidence = _CONF_PARSE_ONLY

        if mode is not Mode.FULL or not self.policy.tests_template:
            return a

        # Free self-check: have the local model turn spec examples into asserts, run them.
        tests_user = f"{self.policy.tests_template}\n\nTask:\n{task.prompt}\n\nCandidate:\n{code}"
        tests_resp = await self.generate(
            tests_user, a, system="You write minimal Python assert tests.", temperature=0.0
        )
        asserts = extract_code_block(strip_thinking(tests_resp.text)) or ""
        if not asserts or asserts.strip() == _NO_TESTS_SENTINEL:
            return a  # nothing checkable in the spec; keep parse-level confidence

        result = await run_asserts(code, asserts)
        if not result.ok and self.can_repair(mode, 1):
            resp = await self.generate(
                self.render_repair(task, code, result.stderr or "assert failed"), a
            )
            fixed = extract_code_or_all(strip_thinking(resp.text))
            if not code_parses(fixed):
                code = fixed
                a.answer = self.format_answer(code)
                a.repaired = True
                result = await run_asserts(code, asserts)
        if result.ok:
            a.confidence = _CONF_TESTS_PASS
        # Failing self-made asserts is a soft signal (the asserts themselves may be wrong):
        # keep the parse-level answer rather than paying for escalation.
        return a


class CodeDebugSolver(BaseSolver):
    """Fix the bug, then prove the fix by running it (against expected output when stated)."""

    async def attempt_local(self, task: Task, mode: Mode) -> LocalAttempt:
        a = LocalAttempt()
        expected = extract_expected_output(task.prompt)
        resp = await self.generate(self.render_user(task), a)
        text = strip_thinking(resp.text)
        code = extract_code_block(text)

        for round_ in range(self.policy.repair_attempts + 1):
            if code is None:
                feedback = "reply contained no fenced code block"
            else:
                result = await run_python(code)
                if not result.ok:
                    feedback = result.stderr or "execution failed"
                elif expected is not None and expected not in result.stdout:
                    feedback = (
                        f"output was {result.stdout[:200]!r} but expected {expected[:200]!r}"
                    )
                else:
                    a.answer = self.format_answer(text)
                    a.verified = True
                    a.confidence = _CONF_RUNS_CLEAN if expected is None else _CONF_TESTS_PASS
                    return a
            if not self.can_repair(mode, round_):
                a.feedback = feedback
                return a
            resp = await self.generate(self.render_repair(task, code or text, feedback), a)
            text = strip_thinking(resp.text)
            code = extract_code_block(text)
            a.repaired = True
        a.feedback = a.feedback or "exhausted repair attempts"
        return a
