"""Interactive one-task pipeline for the dashboard playground.

Runs a typed-in prompt through the exact production stack — dispatcher, solver,
verification, escalation — and reports which route answered and what it cost.
The pipeline (backends, dispatcher, spawned llama-server) initializes lazily on
the first request and lives in a dedicated asyncio thread; each request gets a
fresh Budget so long-lived playground sessions never degrade into panic mode.

Demo-safe defaults come from the same env switches as a run: on a laptop with
mock backends the deterministic lane still answers for real (it is pure Python).
"""

from __future__ import annotations

import asyncio
import atexit
import itertools
import threading

from src.core import settings
from src.core.config import load_config
from src.core.schemas import Category, Route, SolveResult, Task
from src.llm.local_llm import make_local
from src.llm.local_server import ensure_local_server
from src.llm.remote_llm import make_remote
from src.routing.budget import Budget
from src.routing.dispatcher import Dispatcher
from src.solvers import make_solvers
from src.solvers.base import SolveContext
from src.solvers.deterministic import try_deterministic

_SOLVE_TIMEOUT_S = 120.0
_INIT_TIMEOUT_S = 90.0  # may include a llama-server spawn + model load


class PlaygroundError(Exception):
    pass


class Playground:
    """Thread-safe lazy singleton around the pipeline, driven from sync handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ids = itertools.count(1)
        self._config = None
        self._local = None
        self._remote = None
        self._dispatcher = None
        self._server = None

    # ---- sync facade (called from http.server handler threads) ----------------

    def solve(self, prompt: str) -> dict:
        prompt = prompt.strip()
        if not prompt:
            raise PlaygroundError("empty prompt")
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._solve(prompt), loop)
        try:
            return future.result(timeout=_SOLVE_TIMEOUT_S + _INIT_TIMEOUT_S)
        except TimeoutError as e:
            future.cancel()
            raise PlaygroundError("solve timed out") from e

    # ---- async internals -------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, daemon=True,
                                 name="playground-loop").start()
                self._loop = loop
                atexit.register(self._shutdown)
            return self._loop

    async def _init_once(self) -> None:
        if self._config is not None:
            return
        config = load_config()
        self._local = make_local(config.local)
        self._remote = make_remote(config.remote)
        self._dispatcher = Dispatcher()
        self._server = await ensure_local_server(config.local)
        await self._local.wait_ready(settings.LOCAL_READY_TIMEOUT_S)
        if hasattr(self._local, "warmup"):
            await self._local.warmup()
        self._config = config

    async def _solve(self, prompt: str) -> dict:
        async with asyncio.timeout(_INIT_TIMEOUT_S):
            await self._init_once()
        budget = Budget()  # fresh clock: playground tasks always run in FULL mode
        ctx = SolveContext(
            config=self._config, local=self._local, remote=self._remote,
            budget=budget, remaining_tasks=lambda: 1,
        )
        solvers = make_solvers(ctx)
        task = Task(task_id=f"play-{next(self._ids)}", prompt=prompt)
        async with asyncio.timeout(_SOLVE_TIMEOUT_S):
            det = try_deterministic(prompt)
            if det is not None:
                method = "deterministic"
                result = SolveResult(
                    task_id=task.task_id, answer=det, category=Category.MATH,
                    route=Route.DETERMINISTIC, confidence=1.0, detail="exact-match handler",
                )
            else:
                category, method = await self._dispatcher.classify(prompt, self._local)
                result = await solvers[category].solve(task)
        return {
            "answer": result.answer,
            "category": result.category.value,
            "route": result.route.value,
            "dispatch": method,
            "confidence": round(result.confidence, 3),
            "remote_tokens": result.remote_prompt_tokens + result.remote_completion_tokens,
            "local_tokens": result.local_prompt_tokens + result.local_completion_tokens,
            "wall_ms": result.wall_ms,
            "local_ms": result.local_ms,
            "remote_ms": result.remote_ms,
            "detail": result.detail,
        }

    def _shutdown(self) -> None:
        loop, self._loop = self._loop, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if self._server is not None:
            self._server.stop()
