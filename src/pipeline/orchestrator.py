"""Run loop: dispatch -> solve concurrently under budget -> record -> guarantee output.

Guarantees on exit: every task has an answer in results.json, the file is valid JSON,
and at least one Fireworks call was made (AMD-usage rule) via the end-of-run insurance
check if the whole run stayed local.
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

_INSURANCE_CONF = 0.9


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
                result = SolveResult(
                    task_id=task.task_id, answer="", category=Category.FACTUAL,
                    route=Route.ERROR, detail="task deadline exceeded",
                )
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

        await _ensure_remote_usage(tasks, results, ctx, writer, ledger, metrics, budget)
    finally:
        writer.flush()
        metrics.flush(done=True, budget_elapsed_s=budget.elapsed())
        metrics.print_final(budget_elapsed_s=budget.elapsed())
        await local.close()
        await remote.close()
        if server is not None:
            server.stop()


async def _ensure_remote_usage(
    tasks: list[Task],
    results: dict[str, SolveResult],
    ctx: SolveContext,
    writer: ResultsWriter,
    ledger: Ledger,
    metrics: Metrics,
    budget: Budget,
) -> None:
    """Rule: at least one Fireworks call per run. Spend it on the shakiest local answer."""
    if ledger.remote_calls > 0 or not tasks:
        return
    target = min(
        (r for r in results.values()), default=None, key=lambda r: r.confidence,
    )
    if target is None:
        return
    task = next(t for t in tasks if t.task_id == target.task_id)
    policy = ctx.config.policy(target.category)
    try:
        t0 = budget.elapsed()
        resp = await ctx.remote.complete(
            build_escalation_user(task.prompt, policy.escalate.instruction),
            max_tokens=policy.escalate.max_tokens,
            temperature=policy.escalate.temperature,
            category=target.category.value,
        )
        answer = resp.text.strip()
        if answer:
            writer.set(task.task_id, answer)
            writer.flush()
        metrics.add(ledger.record(SolveResult(
            task_id=task.task_id, answer=answer or target.answer,
            category=target.category, route=Route.INSURANCE,
            confidence=_INSURANCE_CONF if answer else target.confidence,
            remote_prompt_tokens=resp.prompt_tokens,
            remote_completion_tokens=resp.completion_tokens,
            remote_calls=1,
            remote_ms=int((budget.elapsed() - t0) * 1000),
            finished_s=budget.elapsed(),
        )))
        log.info("insurance Fireworks call made for task %s", task.task_id)
    except RemoteError as e:
        log.error("insurance remote call failed: %r", e)
