"""Central constants, paths, and environment bindings.

Every tunable lives here or in config/*.yaml — other modules must not read os.environ
directly. Harness variables (FIREWORKS_*) are read lazily via accessors so that imports
never fail in dev/mock mode.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ---- I/O paths (harness contract: /input/tasks.json -> /output/results.json) ----
INPUT_TASKS_PATH = Path(_env("VERDICT_INPUT", "/input/tasks.json"))
OUTPUT_RESULTS_PATH = Path(_env("VERDICT_OUTPUT", "/output/results.json"))
LEDGER_PATH = Path(_env("VERDICT_LEDGER", "/tmp/verdict.ledger.jsonl"))
METRICS_PATH = Path(_env("VERDICT_METRICS", "/tmp/verdict.metrics.json"))

# ---- Local model presets: one-word switch via VERDICT_LOCAL_MODEL_PROFILE ----
# Each preset fills in the weights repo/file and the thinking behaviour. Any explicit
# VERDICT_MODEL_REPO / VERDICT_MODEL_FILE / VERDICT_LOCAL_* always overrides a preset.
# All are ~1.9-2.8 GB Q4 GGUF so they fit the 4 GB CPU grading box (with ctx <= 4096).
LOCAL_MODEL_PROFILES: dict[str, dict[str, str]] = {
    # Primary: best-in-class 4B, natively non-thinking (no <think>), terse by default.
    "qwen3-4b-2507": {
        "VERDICT_MODEL_REPO": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "VERDICT_MODEL_FILE": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "VERDICT_LOCAL_ENABLE_THINKING": "false",
        "VERDICT_LOCAL_SEND_THINK_KWARG": "false",  # native non-thinking; send no kwarg
    },
    # Challenger: math/factual-elite 3.8B (GSM8K 88, MMLU 73), also non-thinking.
    "phi4-mini": {
        "VERDICT_MODEL_REPO": "unsloth/Phi-4-mini-instruct-GGUF",
        "VERDICT_MODEL_FILE": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "VERDICT_LOCAL_ENABLE_THINKING": "false",
        "VERDICT_LOCAL_SEND_THINK_KWARG": "false",
    },
    # Safety net: smaller, comfortable on RAM + 30 s/request if 4B is too tight/slow.
    "qwen25-3b": {
        "VERDICT_MODEL_REPO": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "VERDICT_MODEL_FILE": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "VERDICT_LOCAL_ENABLE_THINKING": "false",
        "VERDICT_LOCAL_SEND_THINK_KWARG": "false",
    },
    # Prior default: a thinking model, forced OFF via the chat-template kwarg.
    "qwen35-4b": {
        "VERDICT_MODEL_REPO": "unsloth/Qwen3.5-4B-GGUF",
        "VERDICT_MODEL_FILE": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
        "VERDICT_LOCAL_ENABLE_THINKING": "false",
        "VERDICT_LOCAL_SEND_THINK_KWARG": "true",
    },
}
DEFAULT_LOCAL_PROFILE = "qwen3-4b-2507"


def _apply_local_profile() -> None:
    """Expand VERDICT_LOCAL_MODEL_PROFILE into per-var env defaults (setdefault, so any
    explicit VERDICT_* still wins). Unknown names fall through to the hard defaults below."""
    name = os.environ.get("VERDICT_LOCAL_MODEL_PROFILE", DEFAULT_LOCAL_PROFILE).strip()
    for key, val in LOCAL_MODEL_PROFILES.get(name, {}).items():
        os.environ.setdefault(key, val)


_apply_local_profile()

# ---- Local model weights (auto-downloaded when absent; Docker bakes them at build) ----
# Defaults below are the last resort if an unknown profile is named; the default profile
# (qwen3-4b-2507) fills these via _apply_local_profile() above.
MODEL_REPO = _env("VERDICT_MODEL_REPO", "unsloth/Qwen3.5-4B-GGUF")
MODEL_FILE = _env("VERDICT_MODEL_FILE", "Qwen3.5-4B-UD-Q4_K_XL.gguf")
MODEL_DIR = Path(_env("VERDICT_MODEL_DIR", str(REPO_ROOT / "models")))
MODEL_LINK = MODEL_DIR / "model.gguf"  # stable path llama-server is pointed at

# ---- Self-managed llama-server (spawned by the app when nothing answers /health) ----
LLAMA_BIN = _env("VERDICT_LLAMA_BIN", "llama-server")  # name on PATH or explicit path
LLAMA_BIN_CANDIDATES = [  # fallbacks probed after PATH
    REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-server",
    Path("/usr/local/bin/llama-server"),
]
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH")  # explicit weights (container)
# The 4 GB CPU grading box sizes the KV cache to the TOTAL ctx, so keep ctx small
# (2048-4096). Quantizing the KV cache (LLAMA_CACHE_TYPE_*=q8_0) buys more RAM headroom
# but the V cache needs flash-attn. The dev GPU box can raise all of these via env.
LLAMA_CTX = _env_int("LLAMA_CTX", 4096)
LLAMA_PARALLEL = _env_int("LLAMA_PARALLEL", 2)  # keep aligned with TASK_CONCURRENCY
LLAMA_THREADS = _env_int("LLAMA_THREADS", os.cpu_count() or 4)
LLAMA_NGL = _env_int("LLAMA_NGL", 99)  # GPU layers; harmless no-op on CPU builds
LLAMA_CACHE_TYPE_K = _env("LLAMA_CACHE_TYPE_K", "")  # e.g. q8_0; empty -> f16 default
LLAMA_CACHE_TYPE_V = _env("LLAMA_CACHE_TYPE_V", "")  # q8_0 needs LLAMA_FLASH_ATTN set
LLAMA_FLASH_ATTN = _env("LLAMA_FLASH_ATTN", "")      # on|off|auto; empty -> not passed
LLAMA_LOG_PATH = Path(_env("VERDICT_LLAMA_LOG", "/tmp/llama-server.log"))

# ---- Config files ----
CONFIG_DIR = Path(_env("VERDICT_CONFIG_DIR", str(REPO_ROOT / "config")))
MODELS_YAML = CONFIG_DIR / "models.yaml"
CATEGORIES_YAML = CONFIG_DIR / "categories.yaml"
CALIBRATION_JSON = CONFIG_DIR / "calibration.json"

# ---- Wall-clock budget (harness: 10 min total, ~25 s/task ceiling) ----
RUN_BUDGET_S = _env_float("VERDICT_RUN_BUDGET_S", 570.0)
TASK_CEILING_S = _env_float("VERDICT_TASK_CEILING_S", 25.0)
FINAL_FLUSH_RESERVE_S = 5.0  # kept back for the last results.json write
EST_TASK_COST_S = 8.0  # prior for budget pressure before observations exist

# pressure = remaining_time / estimated_remaining_cost; below these -> degrade
PRESSURE_REDUCED = 1.5  # shrink n_samples, skip cross-checks
PRESSURE_PANIC = 0.7  # single fast attempt, accept unverified / escalate immediately

# ---- Concurrency ----
# On the 2 vCPU grading box, more than ~2 in-flight local generations only thrash the
# cores and risk the 30 s/request limit; keep this aligned with LLAMA_PARALLEL. The dev
# GPU box can raise it via VERDICT_TASK_CONCURRENCY.
TASK_CONCURRENCY = _env_int("VERDICT_TASK_CONCURRENCY", 2)

# ---- Local server readiness (container must be ready in <=60 s) ----
LOCAL_READY_TIMEOUT_S = _env_float("VERDICT_LOCAL_READY_TIMEOUT_S", 55.0)
LOCAL_READY_POLL_S = 0.5

# ---- Python-exec verification sandbox ----
PYEXEC_TIMEOUT_S = _env_float("VERDICT_PYEXEC_TIMEOUT_S", 4.0)
PYEXEC_MAX_OUTPUT_CHARS = 10_000

# ---- Mock backend knobs (dev on the i3 laptop) ----
MOCK_FAIL_RATE = _env_float("VERDICT_MOCK_FAIL_RATE", 0.0)
MOCK_SEED = _env_int("VERDICT_MOCK_SEED", 1337)
MOCK_LATENCY_S = _env_float("VERDICT_MOCK_LATENCY_S", 0.05)

LOG_LEVEL = _env("VERDICT_LOG_LEVEL", "INFO")

# ---- Harness-injected (lazy: only needed when a real Fireworks call happens) ----
ENV_FIREWORKS_API_KEY = "FIREWORKS_API_KEY"
ENV_FIREWORKS_BASE_URL = "FIREWORKS_BASE_URL"
ENV_ALLOWED_MODELS = "ALLOWED_MODELS"


def fireworks_api_key() -> str:
    return os.environ[ENV_FIREWORKS_API_KEY]


def fireworks_base_url() -> str:
    return os.environ[ENV_FIREWORKS_BASE_URL].rstrip("/")


def allowed_models() -> list[str]:
    raw = os.environ.get(ENV_ALLOWED_MODELS, "")
    return [m.strip() for m in raw.split(",") if m.strip()]
