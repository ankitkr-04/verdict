"""Local inference backends: llama.cpp llama-server (real) and a deterministic mock.

Local tokens score zero — the only cost here is wall-clock, so calls are bounded by
timeouts and slot concurrency, never by token accounting.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from typing import Any, Protocol

import httpx

from src.core import settings
from src.llm.openai_compat import chat_completion
from src.core.parsing import parse_summary_constraint
from src.core.schemas import LLMResponse


class LocalError(Exception):
    pass


class LocalLLM(Protocol):
    no_think_suffix: str

    async def generate(
        self, system: str, user: str, *, temperature: float, max_tokens: int,
        stop: list[str] | None = None, sampling: dict[str, float] | None = None,
    ) -> LLMResponse: ...

    async def wait_ready(self, timeout_s: float) -> None: ...

    async def close(self) -> None: ...


class LlamaLocal:
    """Client for a llama-server process exposing the OpenAI-compatible API."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.base_url = str(cfg.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.model = str(cfg.get("model", "local"))
        self.logprobs = bool(cfg.get("logprobs", True))
        self.no_think_suffix = str(cfg.get("no_think_suffix", ""))
        # Thinking control. A thinking model (Qwen3.5) whose template defaults thinking ON
        # must be forced OFF via chat_template_kwargs, or hidden reasoning crowds out (or
        # truncates) the real answer. Natively non-thinking models (Qwen3-*-Instruct-2507,
        # Phi-4-mini) need no kwarg — set send_think_kwarg=false so we never hand their
        # template a variable it doesn't declare. Honored by both llama.cpp and vLLM.
        self.extra_body: dict[str, Any] = {}
        thinking_on = str(cfg.get("enable_thinking", "false")).strip().lower() in ("1", "true", "yes", "on")
        send_kwarg = str(cfg.get("send_think_kwarg", "true")).strip().lower() in ("1", "true", "yes", "on")
        if send_kwarg and not thinking_on:
            self.extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        # Model-family sampling defaults (Qwen3.5 card: top_p/top_k/min_p/presence_penalty).
        self.sampling = {
            k: float(v) for k, v in (cfg.get("sampling") or {}).items()
            if v is not None and str(v) != ""
        }
        timeout = float(cfg.get("timeout_s", 20))
        slots = int(cfg.get("slots", 4))
        self._sem = asyncio.Semaphore(slots)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))

    async def generate(
        self, system: str, user: str, *, temperature: float, max_tokens: int,
        stop: list[str] | None = None, sampling: dict[str, float] | None = None,
    ) -> LLMResponse:
        async with self._sem:
            try:
                return await chat_completion(
                    self._client, base_url=self.base_url, model=self.model,
                    system=system, user=user, temperature=temperature,
                    max_tokens=max_tokens, stop=stop, logprobs=self.logprobs,
                    sampling={**self.sampling, **(sampling or {})},
                    extra_body=self.extra_body or None,
                )
            except (httpx.HTTPError, KeyError, ValueError) as e:
                raise LocalError(f"local generation failed: {e!r}") from e

    async def wait_ready(self, timeout_s: float) -> None:
        """Poll llama-server /health (lives at server root, not under /v1)."""
        root = self.base_url.removesuffix("/v1")
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            try:
                r = await self._client.get(f"{root}/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if asyncio.get_running_loop().time() > deadline:
                raise LocalError(f"llama-server not ready within {timeout_s:.0f}s")
            await asyncio.sleep(settings.LOCAL_READY_POLL_S)

    async def warmup(self) -> None:
        """Tiny generation to fault mmap'd weights into memory before the run."""
        try:
            await self.generate("", "Say OK.", temperature=0.0, max_tokens=2)
        except LocalError:
            pass  # warmup is best-effort

    async def close(self) -> None:
        await self._client.aclose()


class MockLocal:
    """Deterministic stand-in for dev on hardware that can't run an LLM.

    Produces structurally valid per-category outputs so every solver/verifier path is
    exercisable; VERDICT_MOCK_FAIL_RATE injects garbage to exercise repair/escalation.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.no_think_suffix = ""
        self._rng = random.Random(settings.MOCK_SEED)

    @staticmethod
    def _stable_words(text: str, k: int) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z-]+", text)
        return words[:k] if words else ["mock"]

    def _reply(self, system: str, user: str) -> str:
        sys_l, user_l = system.lower(), user.lower()
        if "assert statements" in sys_l:
            return "```python\npass\n```"
        if "exactly one word from" in user_l:
            m = re.search(r"from:\s*([a-z_,\s]+)", user_l)
            return (m.group(1).split(",")[0].strip() if m else "factual")
        if "print() of the final answer" in sys_l:
            return "```python\nprint(42)\n```"
        if "requested function" in sys_l:
            return "```python\ndef solution(*args, **kwargs):\n    return None\n```"
        if "fix bugs" in sys_l or "corrected code" in sys_l:
            return "The bug is a mock bug.\n```python\nprint('fixed')\n```"
        if "json array" in sys_l:
            caps = re.findall(r"\b[A-Z][a-z]+\b", user)
            name = caps[0] if caps else "Paris"
            return f'[{{"text": "{name}", "type": "PERSON"}}]'
        if "sentiment" in sys_l:
            return "Positive\nThe wording is enthusiastic throughout."
        if "summarize" in sys_l:
            c = parse_summary_constraint(user)
            if c and c.kind == "bullets":
                n = c.n or 3
                return "\n".join(f"- Mock bullet {i + 1}." for i in range(n))
            if c and c.kind == "words":
                n = max(1, min(c.n, 12))
                return " ".join(["mock"] * (n - 1) + ["summary."])
            n = c.n if c and c.kind == "sentences" else 1
            return " ".join(f"Mock summary sentence {i + 1}." for i in range(n))
        if "final:" in sys_l:
            return "Working through the constraints briefly.\nFINAL: Alice"
        words = " ".join(self._stable_words(user, 6))
        return f"Mock factual answer about {words}."

    async def generate(
        self, system: str, user: str, *, temperature: float, max_tokens: int,
        stop: list[str] | None = None, sampling: dict[str, float] | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(settings.MOCK_LATENCY_S)
        corrupt = self._rng.random() < settings.MOCK_FAIL_RATE
        text = "%%% GARBLED MOCK OUTPUT %%%" if corrupt else self._reply(system, user)
        # Stable pseudo-logprob: confident for clean output, poor for corrupted.
        h = int(hashlib.sha1(user.encode()).hexdigest()[:6], 16) / 0xFFFFFF
        mean_lp = -2.5 if corrupt else -0.05 - 0.3 * h
        return LLMResponse(
            text=text,
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
            mean_logprob=mean_lp,
            model="mock-local",
        )

    async def wait_ready(self, timeout_s: float) -> None:
        return

    async def warmup(self) -> None:
        return

    async def close(self) -> None:
        return


def make_local(cfg: dict[str, Any]) -> LlamaLocal | MockLocal:
    backend = str(cfg.get("backend", "llama")).lower()
    if backend == "mock":
        return MockLocal(cfg)
    if backend == "llama":
        return LlamaLocal(cfg)
    raise ValueError(f"unknown local backend: {backend!r}")
