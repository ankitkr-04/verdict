"""Run loop: dispatch -> solve concurrently under budget -> record -> guarantee output.

Guarantees on exit: every task has an answer in results.json and the file is valid JSON.
Zero Fireworks calls is a valid, optimal outcome — a fully-local run that clears the
accuracy gate scores the minimum (zero) remote tokens, so we never spend a token for its
own sake; escalation happens only on a genuine, verified local failure.
"""

from __future__ import annotations

import asyncio
import logging

from src.core import settings
from src.core.config import load_config
from src.core.schemas import Category, Route, SolveResult, Task
from src.llm.local_llm import make_local
from src.llm.local_server import ensure_local_server
from src.llm.remote_llm import RemoteError, make_remote
from src.pipeline.ledger import Ledger
from src.pipeline.metrics import Metrics
from src.pipeline.results_writer import ResultsWriter
from src.routing.budget import Budget
from src.routing.dispatcher import Dispatcher
from src.routing.prompt_compression import build_escalation_user
from src.solvers import make_solvers
from src.solvers.base import SolveContext
from src.solvers.deterministic import try_deterministic

log = logging.getLogger("verdict.orchestrator")

_LAST_RESORT_TEXT = "Unable to determine a reliable answer."


async def _answer_after_deadline(task: Task, ctx: SolveContext, budget: Budget) -> SolveResult:
    """A task that hit its per-task deadline must STILL get an answer — never drop a
    prompt. The 10-min budget is TOTAL and the per-task cap is a safety limit, so on
    timeout we spend one fast, tightly-capped remote call within the time left; only if
    even that can't run do we ship a non-empty placeholder. Remote is ~10x faster than
    CPU-local, so this reliably beats the deadline it was triggered by.
    """
    left = budget.remaining() - settings.FINAL_FLUSH_RESERVE_S
    if left <= 1.5:
        return SolveResult(
            task_id=task.task_id, answer=_LAST_RESORT_TEXT, category=Category.FACTUAL,
            route=Route.ERROR, detail="deadline exceeded; no time left to escalate",
        )
    try:
        t0 = budget.elapsed()
        async with asyncio.timeout(min(10.0, left)):
            resp = await ctx.remote.complete(
                build_escalation_user(task.prompt, "Answer concisely and correctly."),
                max_tokens=100, temperature=0.0, category="factual",
            )
        ans = resp.text.strip()
        return SolveResult(
            task_id=task.task_id, answer=ans or _LAST_RESORT_TEXT, category=Category.FACTUAL,
            route=Route.ESCALATED if ans else Route.ERROR, confidence=0.8 if ans else 0.0,
            remote_calls=1, remote_prompt_tokens=resp.prompt_tokens,
            remote_completion_tokens=resp.completion_tokens,
            remote_ms=int((budget.elapsed() - t0) * 1000),
            detail="local deadline exceeded -> fast remote answer",
        )
    except (TimeoutError, RemoteError) as e:
        return SolveResult(
            task_id=task.task_id, answer=_LAST_RESORT_TEXT, category=Category.FACTUAL,
            route=Route.ERROR, detail=f"deadline exceeded; remote fallback failed: {e!r}",
        )


async def run(tasks: list[Task], writer: ResultsWriter) -> None:
    config = load_config()
    local = make_local(config.local)
    remote = make_remote(config.remote)
    budget = Budget()
    ledger = Ledger(settings.LEDGER_PATH)
    metrics = Metrics(
        settings.METRICS_PATH,
        str(config.local.get("backend", "llama")),
        str(config.remote.get("backend", "fireworks")),
    )
    dispatcher = Dispatcher()

    pending = len(tasks)
    ctx = SolveContext(
        config=config, local=local, remote=remote, budget=budget,
        remaining_tasks=lambda: pending,
    )
    solvers = make_solvers(ctx)
    results: dict[str, SolveResult] = {}
    slots = asyncio.Semaphore(settings.TASK_CONCURRENCY)

    async def run_one(task: Task) -> None:
        nonlocal pending
        t_queued = budget.elapsed()
        async with slots:
            t_start = budget.elapsed()
            dispatch_ms = 0
            try:
                async with asyncio.timeout(budget.task_deadline_s()):
                    # Deterministic answer-lane first: trivial computables (3*9, day-of-week,
                    # vowel count) are answered with zero model calls and never even classify.
                    det = try_deterministic(task.prompt)
                    if det is not None:
                        result = SolveResult(
                            task_id=task.task_id, answer=det, category=Category.MATH,
                            route=Route.DETERMINISTIC, confidence=1.0,
                            detail="exact-match handler",
                        )
                    else:
                        category, method = await dispatcher.classify(task.prompt, local)
                        dispatch_ms = int((budget.elapsed() - t_start) * 1000)
                        result = await solvers[category].solve(task)
                        result.detail = f"dispatch={method}; {result.detail}".strip("; ")
            except TimeoutError:
                # Per-task deadline hit — never drop the prompt: fast remote answer.
                result = await _answer_after_deadline(task, ctx, budget)
            except Exception as e:  # noqa: BLE001 — one task must never sink the run
                log.exception("task %s crashed", task.task_id)
                result = SolveResult(
                    task_id=task.task_id, answer="", category=Category.FACTUAL,
                    route=Route.ERROR, detail=f"crash: {e!r}",
                )
            pending -= 1
            budget.observe_task(budget.elapsed() - t_start)
            result.queue_ms = int((t_start - t_queued) * 1000)
            result.dispatch_ms = dispatch_ms
            result.finished_s = budget.elapsed()
            if not result.mode:
                result.mode = budget.mode(pending).value
            result.wall_ms = result.wall_ms or int((budget.elapsed() - t_start) * 1000)
            results[task.task_id] = result
            writer.set(task.task_id, result.answer)
            writer.flush()
            metrics.add(ledger.record(result))
            metrics.flush(done=False, budget_elapsed_s=budget.elapsed(),
                          remaining_tasks=pending)
            log.info(
                "%s %s route=%s conf=%.2f rtok=%d ltok=%d %dms mode=%s",
                task.task_id, result.category.value, result.route.value,
                result.confidence,
                result.remote_prompt_tokens + result.remote_completion_tokens,
                result.local_prompt_tokens + result.local_completion_tokens,
                result.wall_ms, result.mode,
            )

    server = None
    try:
        # Self-managed local engine: spawns llama-server (downloading weights
        # if absent) unless one is already listening or the backend is mock.
        server = await ensure_local_server(config.local)
        await local.wait_ready(settings.LOCAL_READY_TIMEOUT_S)
        if hasattr(local, "warmup"):
            await local.warmup()

        async with asyncio.TaskGroup() as tg:
            for task in tasks:
                tg.create_task(run_one(task))
    finally:
        writer.flush()
        metrics.flush(done=True, budget_elapsed_s=budget.elapsed())
        metrics.print_final(budget_elapsed_s=budget.elapsed())
        await local.close()
        await remote.close()
        if server is not None:
            server.stop()
