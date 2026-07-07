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

# ---- Local model weights (GPU box; the Docker image bakes them at build time) ----
MODEL_REPO = _env("VERDICT_MODEL_REPO", "unsloth/Qwen3-4B-Instruct-2507-GGUF")
MODEL_FILE = _env("VERDICT_MODEL_FILE", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
MODEL_DIR = Path(_env("VERDICT_MODEL_DIR", str(REPO_ROOT / "models")))
MODEL_LINK = MODEL_DIR / "model.gguf"  # stable path llama-server is pointed at

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
TASK_CONCURRENCY = _env_int("VERDICT_TASK_CONCURRENCY", 4)

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
