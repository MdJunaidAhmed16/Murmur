# 03 — Tech stack

Two columns matter: what you use, and what you get better at. The second is why
the project exists.

## What you use

| Layer | Choice | Why this one |
| --- | --- | --- |
| Pipeline | Hand-rolled asyncio orchestrator | Own the cancellation and barge-in; see decisions below |
| VAD | Silero VAD (ONNX) | Small, fast, runs on CPU alongside everything else |
| STT | faster-whisper `small.en` | Local, good enough, well understood |
| LLM (local) | Qwen2.5-1.5B-Instruct on vLLM | Fits 8GB with room for KV cache; tool-capable |
| LLM (hosted) | Any API model | Same interface, for the always-on demo |
| TTS | Kokoro 82M (ONNX) | Streaming synthesis, better voice than Piper; see decisions |
| Tools | MCP (Python SDK 2.x), in-process | One tool layer the agent consumes |
| Memory store | SQLite + in-memory numpy matrix | One file, no server, survives restarts |
| Embeddings | bge-small-en-v1.5 via fastembed (ONNX, CPU) | ~6 ms per query, inside the 40 ms recall budget |
| Retrieval | Cosine + recency/use boosts, hand-rolled | Sub-millisecond at this scale; see decisions |
| Web | FastAPI + raw PCM over WebSocket | Audio in and out of a browser |
| Deploy | GCP | Hosted backend and the public demo |

## Decisions

**Pipeline: hand-rolled, not Pipecat or LiveKit (day 0).** The module layout in
`01-architecture.md` already describes an orchestrator that owns cancellation, and
structured concurrency under barge-in is one of the skills this project exists to
build. A framework would hide exactly that. Cost: more audio plumbing on day one.
LiveKit also needs a media server, an extra moving part for one user.

**Transport: raw int16 PCM over a WebSocket.** Browser captures at 16 kHz in an
AudioWorklet and sends 512-sample frames (Silero's frame size). Server sends
24 kHz int16 back; a `clear` event empties the browser's playback queue on
barge-in. Chrome/Edge only: Firefox refuses a 16 kHz context fed by a mic stream.

**Platform: everything in WSL2 Ubuntu.** vLLM does not run natively on Windows.
The repo lives in the WSL filesystem, not OneDrive, so `.venv` and model weights
are not synced. vLLM gets its own venv (`.venv-vllm`) because it pins its own torch.

**TTS: Kokoro over Piper.** Kokoro sounds clearly less synthetic, which matters for
a demo a stranger clicks once. It runs on CPU via onnxruntime to keep VRAM for the
LLM. Risk: it may not meet the 200 ms first-audio budget on CPU. If R1 shows that,
options in order: int8 model, GPU execution provider, fall back to Piper.

**Endpointing is speculative (day 7).** On 16 clips, 200 ms of silence cut 7
utterances mid-way and 600 ms cut none, but 600 ms on every turn would eat most of
the budget. So a turn *starts* at 200 ms and *speaks* no earlier than 600 ms; if the
user carries on in between, it is cancelled unheard and turn continuation merges
their words into the next one. Measured in R3.

**GPU TTS: onnxruntime-gpu pinned to 1.26 (day 7).** Kokoro on CPU was 72% of time
to first audio in R1. onnxruntime-gpu 1.27+ is built for CUDA 13 and ships a
`libcudnn.so.9` of the same name as the CUDA 12 one ctranslate2 loads for Whisper;
1.26 is the last CUDA 12 build, so STT and TTS share one set of CUDA 12 wheels. It
replaces the CPU `onnxruntime` package (a uv override) and still provides the CPU
provider that VAD and embeddings use. `MURMUR_TTS_DEVICE` picks the device; the
default stays CPU until R3 measures both. Revisit when ctranslate2 ships for CUDA 13.

**Retrieval: no LlamaIndex (day 4).** One user's store is a few hundred facts. A
numpy dot product over them is sub-millisecond; the query embedding (~6 ms) is the
whole cost. LlamaIndex would add dependencies and per-call overhead inside a 40 ms
budget for no retrieval gain at this size. Revisit only if the store grows by
orders of magnitude.

**Recall is in-process, not an MCP call.** The memory MCP server exists (forget,
recall, remember, history) and the agent consumes it for `forget`, but recall on the
turn path reads the snapshot directly: a protocol hop is not worth paying inside
the budget.

**Small-model guards around the LLM (day 5).** Qwen2.5-1.5B invents facts, judges
contradictions too eagerly, and calls tools it was not asked to. Each failure got a
deterministic guard rather than a bigger prompt: a grounding check on extracted
facts, an embedding-similarity gate (≥ 0.70) before the contradiction judge, and
the `forget` tool offered only on forget-intent words. Measured in R2.

## What you actually get better at

This is the column that goes on your resume and into interview answers.

**Real-time systems under a hard deadline.** Not throughput — tail latency on a
budget you cannot miss. Streaming at every stage, cancellation that works
mid-flight, and knowing which stage to attack first. This transfers directly to
inference and serving work.

**Memory architecture.** Deciding what to persist, consolidating, retrieving under
a latency budget, handling contradiction. There is no settled answer here, which is
exactly why it is good interview material.

**Structured concurrency and cancellation.** Barge-in forces you to get task
groups, cancellation scopes and resource cleanup right. Most engineers have never
had to, and it shows in interviews when they have.

**Local inference economics.** You already know vLLM. This adds the part where the
GPU is shared with STT and TTS and you have to budget VRAM across three models
rather than one.

**MCP as a producer, not a consumer.** You have used MCP tools. Here you write the
servers, including a non-trivial one with real state behind it.

## Where you are weakest going in

Worth naming so you budget for it.

**Audio.** Sample rates, frame sizes, buffer underruns, echo cancellation. This
will eat day one and part of day two, and it will be unglamorous. Budget for it
and do not treat the time as lost.

**Browser audio.** Getting a microphone stream out of a browser and audio back in,
reliably, across devices. WebRTC if the framework gives it to you; a raw WebSocket
of PCM frames if you want to understand it.

**TTS quality versus speed.** The good-sounding models are slow and the fast ones
sound synthetic. You will make a trade here; write down which way you went and why,
because an interviewer will ask.

## VRAM budget

8GB, three models, and a desktop compositor also wants some.

| Component | Approximate | Note |
| --- | --- | --- |
| LLM (1.5B, bf16) | ~3.1 GB | Weights |
| KV cache | ~2.0 GB | Set `gpu_memory_utilization` to leave room |
| STT | ~1.0 GB | Smaller Whisper variants |
| TTS | ~0.5 GB | Piper is tiny |
| Headroom | ~1.0 GB | Display, fragmentation |

If this does not fit, move STT to CPU before shrinking the LLM. Transcription on
CPU is viable; a smaller LLM costs you tool-calling reliability, which is the thing
most likely to break the whole project.

## Deliberately excluded

- **No speaker diarisation.** One user.
- **No telephony.** Browser only.
- **No wake word.** Push to talk, or open mic with VAD.
- **No multi-language on day one.** Get English working end to end first.
- **No fine-tuning on day one.** Stretch goal if days remain — see the roadmap.
- **No Kubernetes.** One box.

## Versions

Pin everything, and record versions in `results/latency.json` alongside the
numbers. Audio and inference libraries change behaviour between minor releases,
and a latency figure without its stack recorded is not reproducible three weeks
later — including by you.
