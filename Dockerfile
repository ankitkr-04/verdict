# syntax=docker/dockerfile:1
# verdict — linux/amd64 image: llama.cpp local engine + Python routing agent.
# Build:  docker buildx build --platform linux/amd64 -t <registry>/verdict:latest --push .
# Harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS at runtime.

# ---------- stage 1: build llama-server (portable: no -march=native on the judging VM) ----------
FROM python:3.12-slim AS llama-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG LLAMA_CPP_REF=master
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp /llama.cpp \
    || git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp
RUN cmake -S /llama.cpp -B /llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
        -DLLAMA_CURL=OFF \
    && cmake --build /llama.cpp/build --target llama-server -j"$(nproc)"

# ---------- stage 2: model weights ----------
# Swap models with build-args, e.g.:
#   --build-arg MODEL_REPO=unsloth/Qwen3-14B-GGUF --build-arg MODEL_FILE=Qwen3-14B-Q4_K_M.gguf
# or override the full MODEL_URL for non-HF sources.
FROM python:3.12-slim AS model
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG MODEL_REPO=unsloth/Qwen3-4B-Instruct-2507-GGUF
ARG MODEL_FILE=Qwen3-4B-Instruct-2507-Q4_K_M.gguf
ARG MODEL_URL=https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}
RUN curl -fL --retry 3 -o /model.gguf "${MODEL_URL}"

# ---------- stage 3: runtime ----------
FROM python:3.12-slim
WORKDIR /app

COPY --from=llama-build /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=model /model.gguf /models/model.gguf

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY main.py ./

# The app manages llama-server itself: finds the binary on PATH, uses the
# baked weights at LLAMA_MODEL_PATH, spawns and reaps the process.
ENV VERDICT_LOCAL_BACKEND=llama \
    VERDICT_REMOTE_BACKEND=fireworks \
    VERDICT_LOCAL_BASE_URL=http://127.0.0.1:8080/v1 \
    LLAMA_MODEL_PATH=/models/model.gguf

ENTRYPOINT ["python", "main.py"]
