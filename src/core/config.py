"""Typed access to config/*.yaml with ${ENV:-default} expansion and defaults merging."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

from src.core import settings
from src.core.schemas import Category

_ENV_PAT = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    def sub(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")

    return _ENV_PAT.sub(sub, text)


def _load_yaml(path) -> dict[str, Any]:
    return yaml.safe_load(_expand_env(path.read_text(encoding="utf-8"))) or {}


@dataclass(slots=True)
class EscalatePolicy:
    instruction: str = ""
    max_tokens: int = 200
    temperature: float = 0.0


@dataclass(slots=True)
class CategoryPolicy:
    name: Category = Category.FACTUAL
    solver: str = "factual"
    verifier: str = "factual_agreement"
    system: str = ""
    user_template: str = "{prompt}"
    repair_template: str = ""
    tests_template: str = ""
    temperature: float = 0.2
    max_tokens: int = 384
    n_samples: int = 1
    repair_attempts: int = 1
    thinking: bool = False
    theta: float = 0.55
    answer_format: str = "{answer}"
    escalate: EscalatePolicy = field(default_factory=EscalatePolicy)
    dispatch_keywords: list[str] = field(default_factory=list)
    dispatch_patterns: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def extra_num(self, key: str, default: float) -> float:
        try:
            return float(self.extra.get(key, default))
        except (TypeError, ValueError):
            return default


_POLICY_KEYS = {
    "solver", "verifier", "prompt", "temperature", "max_tokens", "n_samples",
    "repair_attempts", "thinking", "theta", "answer_format", "escalate", "dispatch",
}


def _build_policy(name: str, raw: dict[str, Any], defaults: dict[str, Any]) -> CategoryPolicy:
    merged: dict[str, Any] = {**defaults, **raw}
    esc_raw = {**defaults.get("escalate", {}), **(raw.get("escalate") or {})}
    prompt = merged.get("prompt") or {}
    dispatch = merged.get("dispatch") or {}
    extra = {k: v for k, v in raw.items() if k not in _POLICY_KEYS}
    return CategoryPolicy(
        name=Category(name),
        solver=str(merged.get("solver", name)),
        verifier=str(merged.get("verifier", "")),
        system=str(prompt.get("system", "")).strip(),
        user_template=str(prompt.get("user", "{prompt}")),
        repair_template=str(prompt.get("repair", "")),
        tests_template=str(prompt.get("tests", "")),
        temperature=float(merged.get("temperature", 0.2)),
        max_tokens=int(merged.get("max_tokens", 384)),
        n_samples=int(merged.get("n_samples", 1)),
        repair_attempts=int(merged.get("repair_attempts", 1)),
        thinking=bool(merged.get("thinking", False)),
        theta=float(merged.get("theta", 0.55)),
        answer_format=str(merged.get("answer_format", "{answer}")),
        escalate=EscalatePolicy(
            instruction=str(esc_raw.get("instruction", "")).strip(),
            max_tokens=int(esc_raw.get("max_tokens", 200)),
            temperature=float(esc_raw.get("temperature", 0.0)),
        ),
        dispatch_keywords=[str(k).lower() for k in dispatch.get("keywords", [])],
        dispatch_patterns=[str(p) for p in dispatch.get("patterns", [])],
        extra=extra,
    )


@dataclass(slots=True)
class Config:
    local: dict[str, Any]
    remote: dict[str, Any]
    categories: dict[Category, CategoryPolicy]
    dispatch_min_score: int
    dispatch_llm_fallback: bool
    dispatch_priority: list[Category]
    calibration: dict[str, dict[str, float]]

    def policy(self, category: Category) -> CategoryPolicy:
        return self.categories[category]

    def platt(self, category: Category) -> tuple[float, float, float]:
        """Return (A, B, theta) for a category, falling back to 'default'."""
        entry = self.calibration.get(category.value) or self.calibration.get("default") or {}
        return (
            float(entry.get("A", -3.0)),
            float(entry.get("B", -0.5)),
            float(entry.get("theta", 0.55)),
        )


@lru_cache(maxsize=1)
def load_config() -> Config:
    models = _load_yaml(settings.MODELS_YAML)
    cats_raw = _load_yaml(settings.CATEGORIES_YAML)
    defaults = cats_raw.get("defaults") or {}
    dispatch = cats_raw.get("dispatch") or {}

    categories: dict[Category, CategoryPolicy] = {}
    for name, raw in (cats_raw.get("categories") or {}).items():
        categories[Category(name)] = _build_policy(name, raw or {}, defaults)
    missing = set(Category) - set(categories)
    if missing:
        raise ValueError(f"categories.yaml missing policies for: {sorted(m.value for m in missing)}")

    calibration: dict[str, dict[str, float]] = {}
    if settings.CALIBRATION_JSON.exists():
        loaded = json.loads(settings.CALIBRATION_JSON.read_text(encoding="utf-8"))
        calibration = {k: v for k, v in loaded.items() if isinstance(v, dict)}

    return Config(
        local=models.get("local") or {},
        remote=models.get("remote") or {},
        categories=categories,
        dispatch_min_score=int(dispatch.get("min_score", 2)),
        dispatch_llm_fallback=bool(dispatch.get("llm_fallback", True)),
        dispatch_priority=[Category(c) for c in dispatch.get("priority", [])],
        calibration=calibration,
    )
