#!/bin/bash
# Gate 3 end to end: installs GPU TTS, measures four configurations, barge-in and tools.
# Needs the GPU to itself. Roughly 20-30 minutes, plus ~0.5 GB of downloads on the first run.
#
#   bash scripts/gate3.sh                                # everything
#   SKIP_SETUP=1 ONLY="baseline clause" bash scripts/gate3.sh   # rerun some measurements
#   uv run python scripts/r3_table.py                    # the table
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
R=results/r3
mkdir -p "$R"

stop_app() {
  for p in $(pgrep -f '\.venv/bin/python -m murmur\.web' || true); do kill "$p"; done
  sleep 3
}
stop_vllm() {
  for p in $(pgrep -f '\.venv-vllm/bin/vllm serve' || true); do kill "$p"; done
}
trap 'stop_app; stop_vllm' EXIT

wait_for() {  # wait_for <file> <pattern> <seconds>
  for _ in $(seq 1 "$3"); do
    grep -qE "$2" "$1" 2>/dev/null && return 0
    sleep 1
  done
  echo "timed out waiting for '$2' in $1" >&2
  tail -20 "$1" >&2
  return 1
}

wanted() {  # ONLY="a b" limits which configurations run
  [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]
}

run_config() {  # run_config <name> <replay args...>; MURMUR_* settings come from the environment
  local name=$1; shift
  wanted "$name" || return 0
  local dir="$R/$name"
  rm -rf "$dir"; mkdir -p "$dir"
  # Settings only; anything named like a secret is left out of the log.
  echo "== $name: $(env | grep '^MURMUR_' | grep -viE 'key|token|secret' | sort | tr '\n' ' ')"
  MURMUR_RESULTS_DIR="$dir" MURMUR_MEMORY_DB="$dir/memory.db" make run > "$dir/app.log" 2>&1 &
  wait_for "$dir/app.log" '"ready"|Traceback' 180
  grep -o '"tts_providers": \[[^]]*\]' "$dir/app.log" || true
  # Read while all three models are resident.
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$dir/vram.txt"
  uv run python scripts/replay.py "$@" > "$dir/replay.txt" 2>&1 || {
    echo "replay failed, see $dir/replay.txt" >&2
  }
  stop_app
  MURMUR_RESULTS_DIR="$dir" uv run python scripts/bench.py > "$dir/bench.txt" 2>&1 || true
  echo "   vram: $(cat "$dir/vram.txt")"
}

# SKIP_SETUP=1 reruns only the measurements (steps 4-6) once install and smoke have passed.
if [ -z "${SKIP_SETUP:-}" ]; then
  echo "== 1. sync: installs onnxruntime-gpu 1.26 (CUDA 12)"
  uv sync

  echo "== 2. TTS smoke, CPU then GPU"
  MURMUR_TTS_DEVICE=cpu make smoke 2>&1 | grep -vi warning | tee "$R/smoke-cpu.txt" \
    || echo "!! CPU smoke failed"
  MURMUR_TTS_DEVICE=cuda make smoke 2>&1 | grep -vi warning | tee "$R/smoke-cuda.txt" \
    || echo "!! GPU smoke failed: the gpu/final configs will fail too; see $R/smoke-cuda.txt"

  echo "== 3. endpointing sweep (VAD only)"
  PYTHONPATH=scripts MURMUR_RESULTS_DIR="$R" uv run python scripts/endpoint_sweep.py \
    clips/*.wav results/gate2-clips/*.wav results/synthetic-clips/0*.wav \
    | tee "$R/endpoint-sweep.txt"
fi

# Time to first audio does not depend on reply length, but the model's 20-60 s answers make
# each run several minutes of playback. Capped for the measurement runs; stated in R3.
export MURMUR_LLM_MAX_TOKENS=96

echo "== 4. vLLM"
make serve > "$R/vllm.log" 2>&1 &
wait_for "$R/vllm.log" 'startup complete|initialization failed' 600

echo "== 5. latency, one change at a time"
MURMUR_TTS_DEVICE=cpu MURMUR_TTS_FIRST_CHUNK=sentence MURMUR_ENDPOINT_COMMIT_MS=0 \
  run_config baseline clips/*.wav
MURMUR_TTS_DEVICE=cpu MURMUR_TTS_FIRST_CHUNK=clause MURMUR_ENDPOINT_COMMIT_MS=0 \
  run_config clause clips/*.wav
MURMUR_TTS_DEVICE=cuda MURMUR_TTS_FIRST_CHUNK=clause MURMUR_ENDPOINT_COMMIT_MS=0 \
  run_config gpu clips/*.wav
MURMUR_TTS_DEVICE=cuda MURMUR_TTS_FIRST_CHUNK=clause MURMUR_ENDPOINT_COMMIT_MS=600 \
  run_config final clips/*.wav
MURMUR_TTS_DEVICE=cuda MURMUR_TTS_FIRST_CHUNK=clause MURMUR_ENDPOINT_COMMIT_MS=600 \
  MURMUR_ENDPOINT_SILENCE_MS=128 run_config tuned clips/*.wav clips/*.wav

echo "== 6. barge-in and tools on the final configuration"
C=results/gate3-clips
MURMUR_TTS_DEVICE=cuda MURMUR_TTS_FIRST_CHUNK=clause MURMUR_ENDPOINT_COMMIT_MS=600 \
  run_config barge-in --interrupt-after 1.5 \
  $C/b1-ask.wav $C/b1-cut.wav $C/b2-ask.wav $C/b2-cut.wav $C/b3-ask.wav $C/b3-cut.wav
MURMUR_TTS_DEVICE=cuda MURMUR_TTS_FIRST_CHUNK=clause MURMUR_ENDPOINT_COMMIT_MS=600 \
  run_config tools $C/t1-time.wav $C/t2-calc.wav $C/t3-calc.wav
if wanted tools; then
  grep -h '"tool_call"' "$R/tools/app.log" | cut -c1-300 || echo "no tool calls logged"
fi

echo "== 7. hosted backend (Claude via OpenRouter), same clips and settings as tuned"
# Only with a key: export MURMUR_OPENROUTER_API_KEY, optionally MURMUR_HOSTED_MODEL.
if [ -n "${MURMUR_OPENROUTER_API_KEY:-}" ]; then
  MURMUR_BACKEND=hosted MURMUR_TTS_DEVICE=cuda MURMUR_TTS_FIRST_CHUNK=clause \
    MURMUR_ENDPOINT_COMMIT_MS=600 MURMUR_ENDPOINT_SILENCE_MS=128 \
    run_config hosted clips/*.wav clips/*.wav
  if wanted hosted; then
    grep -h '"tool_call"' "$R/hosted/app.log" | cut -c1-300 || true
  fi
else
  echo "   skipped: no MURMUR_OPENROUTER_API_KEY"
fi

echo "== done"
uv run python scripts/r3_table.py
