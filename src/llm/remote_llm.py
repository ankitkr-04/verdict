"""Remote (scored) inference: Fireworks AI via the harness-injected environment.

Every call here costs leaderboard tokens. Callers are expected to have minimized the
prompt (see compress.py) and capped max_tokens before reaching this module.
"""

from __future__ import annotations

import asyncio
import json
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
        stop: list[str] | None = None, category: str = "",
    ) -> LLMResponse: ...

    async def close(self) -> None: ...


def _csv(value: Any) -> list[str]:
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(p) for p in (value or [])]


def pick_model(allowed: list[str], prefer: list[str]) -> str:
    """First allowed model matching a preference substring (in order), else first allowed.

    Preferences are decided offline by eval/bakeoff.py and frozen into config; only
    models actually present in ALLOWED_MODELS are ever returned.
    """
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
        self._prefer = _csv(cfg.get("prefer"))
        self._category_prefer = {
            str(cat): _csv(pats)
            for cat, pats in (cfg.get("category_prefer") or {}).items()
            if _csv(pats)
        }
        raw_extra = str(cfg.get("extra_body_json", "")).strip()
        try:
            self._extra_body = json.loads(raw_extra) if raw_extra else None
        except json.JSONDecodeError as e:
            raise ValueError(f"remote.extra_body_json is not valid JSON: {e}") from e
        self._max_retries = int(cfg.get("max_retries", 1))
        self._backoff_s = float(cfg.get("retry_backoff_s", 1.0))
        timeout = float(cfg.get("timeout_s", 25))
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))
        self._by_category: dict[str, str] = {}

    @property
    def model(self) -> str:
        return self.model_for("")

    def model_for(self, category: str) -> str:
        """Resolve (and cache) the model for a category from frozen preferences."""
        if category not in self._by_category:
            prefer = self._category_prefer.get(category, []) + self._prefer
            self._by_category[category] = pick_model(settings.allowed_models(), prefer)
        return self._by_category[category]

    async def complete(
        self, user: str, *, max_tokens: int, temperature: float = 0.0,
        stop: list[str] | None = None, category: str = "",
    ) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await chat_completion(
                    self._client,
                    base_url=settings.fireworks_base_url(),
                    model=self.model_for(category),
                    system=self.system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    api_key=settings.fireworks_api_key(),
                    extra_body=self._extra_body,
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
        stop: list[str] | None = None, category: str = "",
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
