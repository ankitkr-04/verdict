"""Run-level metrics artifact: one JSON file that explains an entire run.

Written atomically after every task completion, so it is always valid and can
be watched live (`watch cat run.metrics.json`) or read after a crash. Consumers:
the future frontend dashboard, and debugging-by-pasting — this file plus the
ledger JSONL is everything needed to diagnose a remote run.

Collection is stdlib-only and best-effort: telemetry must never cost tokens,
add dependencies, or take the run down.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.core import settings

log = logging.getLogger("verdict.metrics")

SCHEMA_VERSION = 1


# ---- hardware snapshot (best-effort, cached once per run) ------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpus() -> list[str]:
    """NVIDIA via nvidia-smi, AMD via rocm-smi; empty list = CPU-only."""
    found: list[str] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            found += [l.strip() for l in out.stdout.splitlines() if l.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("rocm-smi"):
        try:
            out = subprocess.run(
                ["rocm-smi", "--showproductname", "--csv"],
                capture_output=True, text=True, timeout=5,
            )
            found += [l.strip() for l in out.stdout.splitlines()[1:] if l.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    return found


def hardware_snapshot() -> dict:
    return {
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpus": _gpus(),
        "in_docker": Path("/.dockerenv").exists(),
        "python": platform.python_version(),
    }


def config_snapshot(local_backend: str, remote_backend: str) -> dict:
    """Everything env/config-derived that shapes a run's behaviour."""
    return {
        "local_backend": local_backend,
        "remote_backend": remote_backend,
        "model_repo": settings.MODEL_REPO,
        "model_file": settings.MODEL_FILE,
        "llama_model_path": settings.LLAMA_MODEL_PATH,
        "llama_ctx": settings.LLAMA_CTX,
        "llama_parallel": settings.LLAMA_PARALLEL,
        "llama_threads": settings.LLAMA_THREADS,
        "llama_ngl": settings.LLAMA_NGL,
        "task_concurrency": settings.TASK_CONCURRENCY,
        "run_budget_s": settings.RUN_BUDGET_S,
        "task_ceiling_s": settings.TASK_CEILING_S,
        "mock_fail_rate": settings.MOCK_FAIL_RATE,
    }


# ---- aggregation ------------------------------------------------------------------


def _percentiles(values: list[int | float]) -> dict:
    if not values:
        return {"p50": 0, "p95": 0, "max": 0}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"p50": pick(0.50), "p95": pick(0.95), "max": s[-1]}


class Metrics:
    """Accumulates ledger rows and renders/writes the run artifact."""

    def __init__(self, path: Path, local_backend: str, remote_backend: str) -> None:
        self._path = path
        self._t0 = time.time()
        self._rows: list[dict] = []
        self._io_ok = True
        self._hardware = hardware_snapshot()
        self._config = config_snapshot(local_backend, remote_backend)

    def add(self, row: dict) -> None:
        self._rows.append(row)

    # -- derived numbers ---------------------------------------------------------

    def _tokens(self) -> dict:
        rp = sum(r["remote_prompt_tokens"] for r in self._rows)
        rc = sum(r["remote_completion_tokens"] for r in self._rows)
        lp = sum(r["local_prompt_tokens"] for r in self._rows)
        lc = sum(r["local_completion_tokens"] for r in self._rows)
        # What the scored (remote) bill would roughly have been with no local
        # model: one escalation per locally-answered task, prompt ≈ the same
        # prompt we sent locally, completion ≈ what the local model produced.
        est_saved = sum(
            r["local_prompt_tokens"] // max(r["local_calls"], 1) + r["local_completion_tokens"]
            for r in self._rows
            if r["remote_calls"] == 0 and r["local_calls"] > 0
        )
        return {
            "remote_prompt": rp,
            "remote_completion": rc,
            "remote_total": rp + rc,  # <-- the leaderboard number
            "local_prompt": lp,
            "local_completion": lc,
            "local_total": lp + lc,  # free tokens (score zero)
            "est_saved_vs_all_remote": est_saved,  # estimate, see formula above
        }

    def _by_category(self) -> dict:
        cats: dict[str, dict] = {}
        for r in self._rows:
            c = cats.setdefault(r["category"], {
                "tasks": 0, "routes": {}, "remote_tokens": 0,
                "local_tokens": 0, "wall_ms": [],
            })
            c["tasks"] += 1
            c["routes"][r["route"]] = c["routes"].get(r["route"], 0) + 1
            c["remote_tokens"] += r["remote_prompt_tokens"] + r["remote_completion_tokens"]
            c["local_tokens"] += r["local_prompt_tokens"] + r["local_completion_tokens"]
            c["wall_ms"].append(r["wall_ms"])
        for c in cats.values():
            c["wall_ms"] = _percentiles(c.pop("wall_ms"))
        return cats

    def render(self, *, done: bool, budget_elapsed_s: float, remaining_tasks: int) -> dict:
        rows = self._rows
        routes: dict[str, int] = {}
        modes: dict[str, int] = {}
        for r in rows:
            routes[r["route"]] = routes.get(r["route"], 0) + 1
            if r.get("mode"):
                modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if done else "running",
            "run": {
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self._t0)),
                "wall_s": round(budget_elapsed_s, 1),
                "budget_s": settings.RUN_BUDGET_S,
                "headroom_s": round(settings.RUN_BUDGET_S - budget_elapsed_s, 1),
                "tasks_done": len(rows),
                "tasks_remaining": remaining_tasks,
                "modes_seen": modes,  # tasks solved under each budget mode
            },
            "tokens": self._tokens(),
            "routes": routes,
            "latency_ms": {
                "task_wall": _percentiles([r["wall_ms"] for r in rows]),
                "local_gen_per_task": _percentiles(
                    [r["local_ms"] for r in rows if r["local_calls"] > 0]),
                "remote_call": _percentiles(
                    [r["remote_ms"] for r in rows if r["remote_calls"] > 0]),
                "dispatch": _percentiles([r["dispatch_ms"] for r in rows]),
                "queue_wait": _percentiles([r["queue_ms"] for r in rows]),
            },
            "by_category": self._by_category(),
            "hardware": self._hardware,
            "config": self._config,
        }

    # -- output --------------------------------------------------------------------

    def flush(self, *, done: bool, budget_elapsed_s: float, remaining_tasks: int = 0) -> None:
        if not self._io_ok:
            return
        try:
            data = self.render(done=done, budget_elapsed_s=budget_elapsed_s,
                               remaining_tasks=remaining_tasks)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception as e:  # noqa: BLE001 — telemetry never sinks the run
            self._io_ok = False
            log.warning("metrics writes disabled: %r", e)

    def print_final(self, *, budget_elapsed_s: float) -> None:
        try:
            data = self.render(done=True, budget_elapsed_s=budget_elapsed_s,
                               remaining_tasks=0)
            slim = {k: data[k] for k in ("run", "tokens", "routes", "latency_ms")}
            print(json.dumps(slim, indent=2), file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
