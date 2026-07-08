"""Offline model bake-off: probe every allowed Fireworks model, freeze the winners.

Run this OUTSIDE the judged sandbox with your own key (exploratory calls inside the
judged run would bill scored tokens). It sends each probe task's escalation prompt to
every model, measures tokens + latency, runs the free format checks, and prints the
per-category preference lines to freeze into config/models.yaml (or export as env).

Usage (repo root):
    export FIREWORKS_API_KEY=...          # your own dev key, never the harness one
    export ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,..."
    python eval/bakeoff.py [--tasks FILE] [--models a,b,c] [--concurrency 3]

Answers land in outputs/bakeoff/bakeoff.json — read them before trusting the
suggestion; the format checks measure parseability, not truth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import httpx

from src.core import settings
from src.core.config import CategoryPolicy, load_config
from src.core.parsing import extract_code_block, parse_summary_constraint
from src.core.schemas import Category
from src.llm.openai_compat import chat_completion
from src.routing.prompt_compression import build_escalation_user
from src.verification.checks import check_summary, code_parses, validate_entities

DEFAULT_TASKS = REPO_ROOT / "eval" / "mock_tasks" / "tasks.sample.json"
OUT_DIR = REPO_ROOT / "outputs" / "bakeoff"
_TIMEOUT_S = 45.0


def _load_dotenv() -> None:
    """Fill missing env vars from repo-root .env (dev convenience, no dependency)."""
    import os
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


# Self-contained category guess for this offline tool — production routing is now
# LLM-first (no local model here), so bakeoff honors an explicit per-task `category`
# field and otherwise falls back to a tiny keyword map to pick the escalation prompt.
_CATEGORY_HINTS: tuple[tuple[Category, tuple[str, ...]], ...] = (
    (Category.CODE_DEBUG, ("fix the bug", "fix this", "debug", "traceback", "crashes", "why does")),
    (Category.CODE_GEN, ("write a function", "write a python", "implement", "def ", "write a program")),
    (Category.NER, ("named entit", "entities", "extract all", "extract the")),
    (Category.SUMMARIZE, ("summarize", "summarise", "condense", "tl;dr", "one sentence", "in at most")),
    (Category.SENTIMENT, ("sentiment", "positive, negative", "tone of", "how does the reviewer")),
    (Category.LOGIC, ("puzzle", "deduce", "if all", "who sits", "who is", "conclude", "must be true")),
    (Category.MATH, ("calculate", "compute", "how many", "how much", "what is", "%", "percent")),
)


def guess_category(task: dict) -> Category:
    explicit = task.get("category")
    if explicit:
        try:
            return Category(str(explicit))
        except ValueError:
            pass
    text = (task.get("prompt") or "").lower()
    for cat, hints in _CATEGORY_HINTS:
        if any(h in text for h in hints):
            return cat
    return Category.FACTUAL


def format_ok(category: Category, policy: CategoryPolicy, prompt: str, text: str) -> bool:
    """Free parseability check — validates format, NOT truth. Review answers manually."""
    text = text.strip()
    if not text:
        return False
    if category is Category.MATH:
        return len(text) <= 200 and any(ch.isdigit() for ch in text)
    if category is Category.NER:
        try:
            value = json.loads(text[text.index("[") : text.rindex("]") + 1])
        except (ValueError, json.JSONDecodeError):
            return False
        types = [str(t) for t in policy.extra.get("entity_types", [])] or None
        return validate_entities(value, types or ["PERSON", "ORG", "LOCATION", "DATE"]) is None
    if category is Category.SUMMARIZE:
        return check_summary(text, parse_summary_constraint(prompt)) is None
    if category in (Category.CODE_GEN, Category.CODE_DEBUG):
        code = extract_code_block(text) or (text if category is Category.CODE_GEN else None)
        return code is not None and code_parses(code) is None
    if category is Category.SENTIMENT:
        labels = [str(l).lower() for l in policy.extra.get("labels", [])]
        first = text.splitlines()[0].lower()
        return any(l in first for l in labels) if labels else True
    return True  # factual / logic: only manual review can judge these


async def probe(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, *, model: str, task: dict,
    category: Category, policy: CategoryPolicy, system: str,
) -> dict[str, Any]:
    user = build_escalation_user(task["prompt"], policy.escalate.instruction)
    row: dict[str, Any] = {
        "model": model, "task_id": task["task_id"], "category": category.value,
    }
    async with sem:
        t0 = time.monotonic()
        try:
            resp = await chat_completion(
                client, base_url=settings.fireworks_base_url(), model=model,
                system=system, user=user, temperature=policy.escalate.temperature,
                max_tokens=policy.escalate.max_tokens,
                api_key=settings.fireworks_api_key(),
            )
        except (httpx.HTTPError, KeyError, ValueError) as e:
            row.update(error=repr(e), ms=int((time.monotonic() - t0) * 1000))
            return row
    row.update(
        ms=int((time.monotonic() - t0) * 1000),
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        total_tokens=resp.total_tokens,
        format_ok=format_ok(category, policy, task["prompt"], resp.text),
        answer=resp.text,
    )
    return row


def suggest(rows: list[dict], models: list[str]) -> dict[str, str]:
    """Per category: the model with the best format-pass rate, tokens as tie-break."""
    picks: dict[str, str] = {}
    categories = sorted({r["category"] for r in rows})
    for cat in categories:
        stats = []
        for model in models:
            sub = [r for r in rows if r["model"] == model and r["category"] == cat]
            done = [r for r in sub if "error" not in r]
            if not done:
                continue
            ok_rate = sum(r["format_ok"] for r in done) / len(done)
            avg_tok = statistics.mean(r["total_tokens"] for r in done)
            stats.append((-ok_rate, avg_tok, model))
        if stats:
            stats.sort()
            picks[cat] = stats[0][2]
    return picks


async def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--models", default="", help="comma list; default: $ALLOWED_MODELS")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or settings.allowed_models()
    if not models:
        print("no models: set ALLOWED_MODELS or pass --models", file=sys.stderr)
        return 2
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    config = load_config()
    system = str(config.remote.get("system", "")).strip()

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S, connect=10.0)) as client:
        jobs = []
        for task in tasks:
            category = guess_category(task)
            policy = config.policy(category)
            for model in models:
                jobs.append(probe(client, sem, model=model, task=task,
                                  category=category, policy=policy, system=system))
        rows = list(await asyncio.gather(*jobs))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "bakeoff.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'model':<42} {'calls':>5} {'errs':>4} {'fmt-ok':>6} {'avg-tok':>8} {'p95-ms':>7}")
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        done = [r for r in sub if "error" not in r]
        errs = len(sub) - len(done)
        ok = f"{sum(r['format_ok'] for r in done) / len(done):.0%}" if done else "-"
        tok = f"{statistics.mean(r['total_tokens'] for r in done):.0f}" if done else "-"
        lat = f"{sorted(r['ms'] for r in done)[max(0, int(0.95 * len(done)) - 1)]}" if done else "-"
        print(f"{model:<42} {len(sub):>5} {errs:>4} {ok:>6} {tok:>8} {lat:>7}")

    picks = suggest(rows, models)
    print(f"\nanswers: {out}  (READ THEM — format checks measure parseability, not truth)")
    print("\nsuggested freeze (paste into config/models.yaml category_prefer, or export):")
    for cat, model in sorted(picks.items()):
        print(f"  export VERDICT_PREFER_{cat.upper()}={model}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
