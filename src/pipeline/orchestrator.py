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
from src.pipeline.results_writer import ResultsWriter
from src.routing.budget import Budget
from src.routing.dispatcher import Dispatcher
from src.routing.prompt_compression import build_escalation_user
from src.solvers import make_solvers
from src.solvers.base import SolveContext

log = logging.getLogger("verdict.orchestrator")

_INSURANCE_CONF = 0.9


async def run(tasks: list[Task], writer: ResultsWriter) -> None:
    config = load_config()
    local = make_local(config.local)
    remote = make_remote(config.remote)
    budget = Budget()
    ledger = Ledger(settings.LEDGER_PATH)
    dispatcher = Dispatcher(config)

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
        async with slots:
            t_start = budget.elapsed()
            try:
                async with asyncio.timeout(budget.task_deadline_s()):
                    category, method = await dispatcher.classify(task.prompt, local)
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
            results[task.task_id] = result
            writer.set(task.task_id, result.answer)
            writer.flush()
            ledger.record(result)

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

        await _ensure_remote_usage(tasks, results, ctx, writer, ledger)
    finally:
        writer.flush()
        ledger.print_summary()
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
        resp = await ctx.remote.complete(
            build_escalation_user(task.prompt, policy.escalate.instruction),
            max_tokens=policy.escalate.max_tokens,
            temperature=policy.escalate.temperature,
        )
        answer = resp.text.strip()
        if answer:
            writer.set(task.task_id, answer)
            writer.flush()
        ledger.record(SolveResult(
            task_id=task.task_id, answer=answer or target.answer,
            category=target.category, route=Route.INSURANCE,
            confidence=_INSURANCE_CONF if answer else target.confidence,
            remote_prompt_tokens=resp.prompt_tokens,
            remote_completion_tokens=resp.completion_tokens,
            remote_calls=1,
        ))
        log.info("insurance Fireworks call made for task %s", task.task_id)
    except RemoteError as e:
        log.error("insurance remote call failed: %r", e)
