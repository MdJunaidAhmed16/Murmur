# The public demo: hosted LLM (Claude via OpenRouter), Whisper and Kokoro on the container's GPU.
# Built by Cloud Build (scripts/deploy_cloudrun.sh), so model downloads happen on Google's
# network, not the developer's.
FROM python:3.12-slim-bookworm

# ctranslate2 and onnxruntime need OpenMP; everything CUDA comes from the nvidia-* pip wheels,
# and the driver is provided by the Cloud Run GPU host.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first: this layer only rebuilds when the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV MURMUR_BACKEND=hosted \
    MURMUR_DEMO_MODE=true \
    MURMUR_HOST=0.0.0.0 \
    MURMUR_PORT=8080 \
    MURMUR_MODELS_DIR=/app/models \
    MURMUR_DEMO_DIR=/tmp/visitors \
    MURMUR_RESULTS_DIR=/tmp/results \
    MURMUR_MEMORY_DB=/tmp/memory.db

# Weights baked into the image: a cold start loads from local disk instead of downloading.
# Two Whisper sizes: small.en for a GPU instance, base.en for CPU-only, where small.en took
# 1.3 s for one sentence on 4 cores. The deploy picks one with MURMUR_STT_MODEL.
COPY scripts/download_models.py ./scripts/
RUN uv run --no-sync python scripts/download_models.py \
    && MURMUR_STT_MODEL=base.en uv run --no-sync python scripts/download_models.py

# ctranslate2 finds cuBLAS/cuDNN from the pip wheels only if they are on the loader path.
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib

EXPOSE 8080
CMD ["uv", "run", "--no-sync", "python", "-m", "murmur.web"]
