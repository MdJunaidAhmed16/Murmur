.PHONY: setup setup-vllm models serve run smoke bench test lint typecheck check

LLM_MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
# Leaves ~3 GB for Whisper, Kokoro and the desktop (docs/03 VRAM budget).
VLLM_GPU_UTIL ?= 0.55
VLLM_MAX_LEN ?= 4096

# ctranslate2 finds cuBLAS/cuDNN from the pip wheels only if they are on the loader path.
CUDA_LIBS = $(shell .venv/bin/python -c "import nvidia.cublas, nvidia.cudnn; print(nvidia.cublas.__path__[0] + '/lib:' + nvidia.cudnn.__path__[0] + '/lib')" 2>/dev/null)

setup:
	uv sync
	uv run pre-commit install

# vLLM pins its own torch; a separate venv keeps it from fighting the app's dependencies.
setup-vllm:
	uv venv .venv-vllm --python 3.12
	VIRTUAL_ENV=.venv-vllm uv pip install vllm

models:
	uv run python scripts/download_models.py

# WSL appends Windows PATH entries; an unreadable one makes torch.compile's nvcc probe raise
# PermissionError instead of "not found", which kills vLLM at startup. Linux entries only.
LINUX_PATH = $(shell echo "$$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)

# FlashInfer's sampler JIT-compiles with nvcc + ninja (full CUDA toolkit); the torch sampler doesn't.
serve:
	PATH="$(LINUX_PATH)" VLLM_USE_FLASHINFER_SAMPLER=0 .venv-vllm/bin/vllm serve $(LLM_MODEL) \
		--gpu-memory-utilization $(VLLM_GPU_UTIL) \
		--max-model-len $(VLLM_MAX_LEN) \
		--enable-auto-tool-choice --tool-call-parser hermes \
		--port 8000

run:
	LD_LIBRARY_PATH=$(CUDA_LIBS):$$LD_LIBRARY_PATH uv run python -m murmur.web

smoke:
	LD_LIBRARY_PATH=$(CUDA_LIBS):$$LD_LIBRARY_PATH uv run python scripts/smoke.py

bench:
	uv run python scripts/bench.py

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:
	uv run mypy

check: lint typecheck test
