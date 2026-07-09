"""Self-managed llama-server: detect, fetch weights, spawn, reap.

The app owns the local engine end-to-end. At startup: if something already
answers at the configured base_url (a manually started server, any state),
leave it alone. Otherwise ensure the GGUF exists (auto-download when absent)
and spawn llama-server as a child process, killed when the run ends. One code
path for the GPU box and the container — no wrapper scripts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.core import settings
from src.llm.model_fetch import ensure_model

log = logging.getLogger("verdict.local_server")


class LocalServerError(Exception):
    pass


class LocalServer:
    """Handle for a llama-server child we spawned (and must reap)."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        log.info("llama-server pid=%d stopped", self._proc.pid)


async def _something_listening(base_url: str) -> bool:
    """Any HTTP response counts — a server still loading its model answers 503."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(url)
        return True
    except httpx.HTTPError:
        return False


def _find_binary() -> str | None:
    name = settings.LLAMA_BIN
    if os.path.sep in name:  # explicit path via VERDICT_LLAMA_BIN
        return name if os.access(name, os.X_OK) else None
    if found := shutil.which(name):
        return found
    for cand in settings.LLAMA_BIN_CANDIDATES:
        if os.access(cand, os.X_OK):
            return str(cand)
    return None


def _weights_path() -> Path:
    """Explicit LLAMA_MODEL_PATH wins when present; else download-if-absent."""
    if settings.LLAMA_MODEL_PATH:
        explicit = Path(settings.LLAMA_MODEL_PATH)
        if explicit.exists():
            return explicit
        log.warning("LLAMA_MODEL_PATH=%s missing — falling back to fetch", explicit)
    return ensure_model()


async def ensure_local_server(local_cfg: dict[str, Any]) -> LocalServer | None:
    """Return a handle if we spawned the server; None if external/mock."""
    if str(local_cfg.get("backend", "llama")).lower() != "llama":
        return None
    base_url = str(local_cfg.get("base_url", "http://127.0.0.1:8080/v1")).rstrip("/")
    if await _something_listening(base_url):
        log.info("using externally managed llama-server at %s", base_url)
        return None

    binary = _find_binary()
    if binary is None:
        raise LocalServerError(
            f"nothing answers at {base_url} and no llama-server binary found "
            f"(install one on PATH or set VERDICT_LLAMA_BIN)"
        )
    model = await asyncio.to_thread(_weights_path)

    parts = urlsplit(base_url)
    cmd = [
        binary,
        "--model", str(model),
        "--host", parts.hostname or "127.0.0.1",
        "--port", str(parts.port or 8080),
        "--ctx-size", str(settings.LLAMA_CTX),
        "--parallel", str(settings.LLAMA_PARALLEL),
        "--threads", str(settings.LLAMA_THREADS),
        "--n-gpu-layers", str(settings.LLAMA_NGL),
        "--no-webui",
    ]
    # Optional KV-cache quantization + flash-attn for RAM headroom on the 4 GB box.
    if settings.LLAMA_CACHE_TYPE_K:
        cmd += ["--cache-type-k", settings.LLAMA_CACHE_TYPE_K]
    if settings.LLAMA_CACHE_TYPE_V:
        cmd += ["--cache-type-v", settings.LLAMA_CACHE_TYPE_V]
    if settings.LLAMA_FLASH_ATTN:
        cmd += ["--flash-attn", settings.LLAMA_FLASH_ATTN]
    settings.LLAMA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logfile = open(settings.LLAMA_LOG_PATH, "ab")  # noqa: SIM115 — outlives this scope
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=logfile)
    # Fail fast on instant crashes (missing shared libs, bad flags): otherwise the app
    # silently polls a dead port until the readiness timeout with no diagnosis.
    await asyncio.sleep(1.5)
    if proc.poll() is not None:
        try:
            tail = settings.LLAMA_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-500:]
        except OSError:
            tail = "<no log>"
        raise LocalServerError(
            f"llama-server exited immediately (code {proc.returncode}); log tail:\n{tail}"
        )
    log.info("spawned llama-server pid=%d (log: %s)", proc.pid, settings.LLAMA_LOG_PATH)
    return LocalServer(proc)
