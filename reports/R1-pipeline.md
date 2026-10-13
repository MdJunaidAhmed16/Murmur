# R1 — Pipeline

Gate 1: it holds a spoken conversation, and time to first audio is measured.

> Every latency figure here is from `results/latency.json` (`make bench`, 2026-10-13).
> Nothing is typed from memory.

## Setup

| | |
| --- | --- |
| Machine | RTX 4060 Laptop 8 GB, WSL2 Ubuntu 26.04, 7.4 GB RAM to WSL |
| LLM | Qwen2.5-1.5B-Instruct, vLLM 0.31.0, `gpu_memory_utilization=0.55`, bf16 |
| STT | faster-whisper 1.2.1 `small.en`, float16 on GPU |
| TTS | Kokoro v1.0 ONNX (kokoro-onnx 0.6.1), CPU |
| Handoff | Sentence-level (`MURMUR_TTS_HANDOFF=sentence`) |
| Endpointing | 200 ms silence |

**Voice data.** No recordings of the developer yet. Stand-in: 10 utterances of real
callers from MInDS-14 (en-US, CC-BY-4.0; see `clips/README.md`). 8 kHz phone audio
with fillers and pauses — harder than a laptop mic.

**Turns.** 27 recorded: 15 from `scripts/replay.py` (WAV over the WebSocket), 12 from
headless Chrome with the same clips as a fake microphone, exercising the real browser
capture and playback worklets. 5 were cut by barge-in and excluded; **22 measured**.

## It holds a conversation

Yes. Speech in the browser → transcript → spoken reply, over multiple turns with
working memory. Barge-in stops browser playback and marks the reply interrupted.

## Where the time goes

| Stage | p50 | p90 | Budget |
| --- | --- | --- | --- |
| Endpointing | 195 ms | 202 ms | 200 ms |
| Transcription (final) | 149 ms | 256 ms | 100 ms |
| Model, first token | 50 ms | 65 ms | 250 ms |
| Model, first sentence | 338 ms | 444 ms | — (not in the table) |
| Speech, first audio | **1868 ms** | **3357 ms** | 200 ms |
| **Time to first audio** | **2602 ms** | **4170 ms** | **800 ms** |

`time_to_first_audio` runs from the user's last speech frame to the first TTS chunk
leaving the server; browser buffering is excluded (it belongs in the 50 ms slack).
Memory recall does not exist yet.

**TTS is 72% of time to first audio at p50.** Kokoro on CPU synthesises the whole
first sentence before yielding anything, so its cost scales with sentence length —
hence the wide p50–p90 spread. It is also slower than real time on long sentences:
the browser test heard a 0.5 s gap between sentence one and two (playback underrun).

**`llm_first_sentence` was invisible in the budget.** First token is fast (50 ms, warm,
prefix caching), but TTS cannot start until a sentence boundary: another 338 ms.
Added as a stage after the first run showed ~430 ms unaccounted for.

**STT final is over budget** but not by much; Whisper re-transcribes the whole
utterance at the endpoint. p90 is driven by long and filler-only utterances ("Um…"
took 409–419 ms — Whisper struggles on near-silence).

**Endpointing is on budget but wrong.** 4 of 10 clips were split mid-utterance at a
filler or pause; the agent answered half a sentence and was then barged in on. This
is the "too aggressive" failure mode from `docs/01`, and 200 ms is too low for real
speech. Tuning is day 7; the other failure mode (sluggishness) is not measured yet.

## VRAM with all three models loaded

5129 / 8188 MiB (`nvidia-smi`, app idle after warm-up). vLLM: weights 2.98 GiB, KV
cache 0.38 GiB (14,080 tokens — ample for one user). Whisper on GPU, Kokoro on CPU.
WSL2 RAM is the tighter limit: ~3.1 GB available with everything loaded.

## Bugs found by testing

- **Reply audio leaked over the user's next sentence.** Audio is sent faster than real
  time, so the turn ended (server-side) while the browser was still playing; speech
  then started a new turn without clearing playback. Fixed: speech start clears the
  browser whenever audio was sent since the last clear. Regression test added.
- **vLLM would not start under WSL** (Windows `PATH` entries broke torch.compile's
  `nvcc` probe; FlashInfer's sampler needs the CUDA toolkit). Fixed in the Makefile.

## Not yet right

- **Spoken vs generated text.** On barge-in, history records sentences *sent*, not
  sentences *heard*. Needs playback acks from the browser (day 7).
- **The model ignores the brevity prompt.** Replies run to 15–65 s of audio and
  sometimes contain markdown lists, which are read aloud. A 1.5B model limitation;
  options: `max_tokens`, stripping markdown before TTS, a stronger prompt.

## What to attack first

1. **TTS first audio** (1868 ms → 200 ms budget). In order: run Kokoro on the GPU
   (onnxruntime-gpu; ~3 GB VRAM free), split the first sentence at a clause boundary
   (comma) so the first synthesis is short, try the int8 model. If none gets close,
   fall back to Piper per `docs/03`.
2. **Endpointing** — raise the silence threshold and measure both failure modes.
3. **First-sentence latency** — prompt for a short opening sentence; clause-level
   handoff (item 1) also cuts this.
