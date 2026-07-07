"""Minimal OpenAI-compatible /chat/completions call shared by local and remote clients."""

from __future__ import annotations

from typing import Any

import httpx

from src.core.schemas import LLMResponse


def _mean_logprob(choice: dict[str, Any]) -> float | None:
    content = ((choice.get("logprobs") or {}).get("content")) or []
    values = [t["logprob"] for t in content if isinstance(t, dict) and "logprob" in t]
    if not values:
        return None
    return sum(values) / len(values)


async def chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None = None,
    logprobs: bool = False,
    api_key: str | None = None,
) -> LLMResponse:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop[:4]
    if logprobs:
        payload["logprobs"] = True

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content")) or ""
    usage = data.get("usage") or {}
    return LLMResponse(
        text=text,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        mean_logprob=_mean_logprob(choice),
        model=str(data.get("model", model)),
    )
