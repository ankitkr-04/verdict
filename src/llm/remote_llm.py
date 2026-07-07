"""Remote (scored) inference: Fireworks AI via the harness-injected environment.

Every call here costs leaderboard tokens. Callers are expected to have minimized the
prompt (see compress.py) and capped max_tokens before reaching this module.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx

from src.core import settings
from src.llm.openai_compat import chat_completion
from src.core.schemas import LLMResponse


class RemoteError(Exception):
    pass


class RemoteLLM(Protocol):
    system: str

    async def complete(
        self, user: str, *, max_tokens: int, temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> LLMResponse: ...

    async def close(self) -> None: ...


def pick_model(allowed: list[str], prefer: list[str]) -> str:
    """First allowed model matching a preference substring (in order), else first allowed."""
    if not allowed:
        raise RemoteError("ALLOWED_MODELS is empty — cannot select a remote model")
    for want in prefer:
        for model in allowed:
            if want.lower() in model.lower():
                return model
    return allowed[0]


class FireworksRemote:
    """All traffic goes through FIREWORKS_BASE_URL with FIREWORKS_API_KEY (rule #5)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.system = str(cfg.get("system", "")).strip()
        self._prefer = [str(p) for p in cfg.get("prefer", [])]
        self._max_retries = int(cfg.get("max_retries", 1))
        self._backoff_s = float(cfg.get("retry_backoff_s", 1.0))
        timeout = float(cfg.get("timeout_s", 25))
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))
        self._model: str | None = None

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = pick_model(settings.allowed_models(), self._prefer)
        return self._model

    async def complete(
        self, user: str, *, max_tokens: int, temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await chat_completion(
                    self._client,
                    base_url=settings.fireworks_base_url(),
                    model=self.model,
                    system=self.system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    api_key=settings.fireworks_api_key(),
                )
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_s * (attempt + 1))
        raise RemoteError(f"fireworks call failed after retries: {last!r}") from last

    async def close(self) -> None:
        await self._client.aclose()


class MockRemote:
    """Offline stand-in; mirrors token accounting so the ledger pipeline is testable."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.system = str(cfg.get("system", "")).strip()
        self.model = "mock-remote"

    async def complete(
        self, user: str, *, max_tokens: int, temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(settings.MOCK_LATENCY_S)
        text = "MOCK-REMOTE-ANSWER"
        return LLMResponse(
            text=text,
            prompt_tokens=(len(self.system) + len(user)) // 4,
            completion_tokens=min(max_tokens, len(text) // 4 + 1),
            model=self.model,
        )

    async def close(self) -> None:
        return


def make_remote(cfg: dict[str, Any]) -> FireworksRemote | MockRemote:
    backend = str(cfg.get("backend", "fireworks")).lower()
    if backend == "mock":
        return MockRemote(cfg)
    if backend == "fireworks":
        return FireworksRemote(cfg)
    raise ValueError(f"unknown remote backend: {backend!r}")
