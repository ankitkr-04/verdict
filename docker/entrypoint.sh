#!/bin/sh
# Boot llama-server in the background, run the agent, propagate its exit code.
# Python owns readiness (polls /health) and the wall-clock budget.
set -u

: "${LLAMA_MODEL_PATH:=/models/model.gguf}"
: "${LLAMA_PORT:=8080}"
: "${LLAMA_CTX:=16384}"
: "${LLAMA_PARALLEL:=4}"
: "${LLAMA_THREADS:=$(nproc)}"

llama-server \
    --model "${LLAMA_MODEL_PATH}" \
    --host 127.0.0.1 --port "${LLAMA_PORT}" \
    --ctx-size "${LLAMA_CTX}" \
    --parallel "${LLAMA_PARALLEL}" \
    --threads "${LLAMA_THREADS}" \
    --no-webui &
SERVER_PID=$!

python main.py
EXIT_CODE=$?

kill "${SERVER_PID}" 2>/dev/null || true
exit "${EXIT_CODE}"
