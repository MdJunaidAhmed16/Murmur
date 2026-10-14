# R3 — Latency

Gate 3: barge-in cuts audio within 200 ms, the partial turn is recorded as spoken, and time
to first audio is inside 800 ms at p50.

> Figures come from `results/r3/*/latency.json` and `turns.jsonl` (`scripts/gate3.sh`,
> `scripts/r3_table.py`), measured 2026-10-14 on an RTX 4060 Laptop (8 GB), WSL2.
> Stack: faster-whisper 1.2.1, ctranslate2 4.8.3, kokoro-onnx 0.6.1, onnxruntime-gpu 1.26.0,
> vLLM 0.31.0, Qwen2.5-1.5B-Instruct. Replies capped at 96 tokens for these runs: time to
> first audio does not depend on reply length, but 60-second answers made each run slow.

## Gate 3

| Check | Result |
| --- | --- |
| Barge-in cuts audio within 200 ms | **Pass.** 65, 66, 66 ms from the interruption's first speech frame to the stop command |
| Partial turn recorded as spoken | **Pass.** Cut 1.5 s into replies; 23, 19 and 14 characters committed, not the full reply |
| Time to first audio ≤ 800 ms at p50 | **Pass, narrowly.** 756 ms p50 on 20 turns (tuned config) |

Why "narrowly": two runs of the same effective configuration
(`gpu`, `final`) gave 789 and 891 ms on 10 turns each — pooled, 840 ms. p50 on 10 turns
moves by ±50 ms between runs. The tuned configuration closes most of that gap rather than
relying on a lucky run.

## Before and after, per stage

Same 10 MInDS-14 clips (`clips/`, real callers, 8 kHz phone audio), local backend.
p50 / p90 ms. Barge-in turns excluded.

| Stage | baseline (≈ R1) | + clause first chunk | + GPU TTS | + 600 ms hold | **tuned: speculate at 128 ms** |
| --- | --- | --- | --- | --- | --- |
| Endpointing | 195 / 196 | 196 / 196 | 195 / 196 | 196 / 196 | **130 / 131** |
| Transcription (final) | 124 / 146 | 128 / 150 | 128 / 136 | 135 / 140 | 128 / 147 |
| Memory recall | 10 / 13 | 8 / 13 | 10 / 11 | 10 / 13 | 10 / 12 |
| Model, first token | 74 / 107 | 77 / 116 | 79 / 123 | 78 / 108 | 118 / 160 |
| Model, first chunk | 328 / 443 | 190 / 316 | 162 / 342 | 231 / 436 | 170 / 338 |
| Speech, first audio | 1712 / 2844 | 1099 / 2403 | **212 / 306** | 267 / 332 | 204 / 352 |
| **Time to first audio** | **2382 / 3722** | 1683 / 3197 | 789 / 1002 | 891 / 1137 | **756 / 1110** |
| Answered turns (n) | 10 | 10 | 10 | 10 | 20 |

VRAM with all three models resident: 5688 MiB (TTS on CPU) → 6228–6304 MiB (TTS on GPU),
of 8188.

### What each change bought

- **Clause-level first chunk: −700 ms p50.** First audio waits for the whole first chunk to
  synthesise, so cutting a long opening sentence at its first comma shortened the slowest
  stage on CPU from 1712 to 1099 ms, and "first chunk" generation from 328 to 190 ms.
- **GPU TTS: −890 ms p50.** Speech, first audio 1099 → 212 ms. The single biggest change.
  Smoke test: a 4.1 s sentence's first chunk took 1129 ms on CPU and 305 ms on GPU, with
  Whisper still working beside it on the shared CUDA 12 libraries (no cuDNN clash).
- **600 ms hold: no measurable cost here.** It never bound on these clips — the fastest
  answer without it took 610 ms. The 789 → 891 difference is run-to-run noise, not the hold.
- **Speculating at 128 ms: −65 ms endpointing,** and the best p50. It also costs: 12 turns
  started early and were cancelled unheard (5 at 200 ms), and the model's first token rose
  79 → 118 ms, plausibly cancelled requests still holding vLLM while the real one starts.

Where the remaining ~750 ms goes (tuned, p50): endpoint 130, transcription 128, recall 10,
first token 118, first chunk 170, first audio 204.

## Endpointing: both failure modes

Measured with the VAD alone (`scripts/endpoint_sweep.py`), 16 single-utterance clips.

| Silence threshold | Utterances cut mid-way | Added to every turn |
| --- | --- | --- |
| 200 ms | 7 / 16 | 200 ms |
| 300 ms | 6 / 16 | 300 ms |
| 400 ms | 4 / 16 | 400 ms |
| 500 ms | 2 / 16 | 500 ms |
| 600 ms | 0 / 16 | 600 ms |

Longest pause inside an utterance: 544 ms. Neither end fits an 800 ms budget.

**Where it landed: speculative endpointing.** The turn *starts* at 128 ms of silence; its
first audio is *held* until 600 ms. In 62 measured turns across the tuned, gpu and final
runs, **no reply was ever heard over a user who had only paused**: every early start
(22) was cancelled before its first sample, and turn continuation merged the user's words
into the next turn. The aggressive setting's cost is wasted GPU work, not a broken
conversation.

## Barge-in

Scripted with `replay.py --interrupt-after 1.5`: ask for something long, interrupt 1.5 s
into the reply. The virtual player stands in for the browser, including the playback report.

| Clip | Heard before the cut | Onset → stop |
| --- | --- | --- |
| "Tell me a long story about a fox…" / "Okay, stop, that's enough." | 23 chars ("Once upon a time, there") | 66 ms |
| "Explain how a car engine works…" / "Wait, actually, never mind." | 19 chars | 66 ms |
| "Describe the history of the Roman empire" / "Hold on, let me ask something else." | 14 chars | 65 ms |

`barge_in` runs from the interruption's first speech frame to the server telling the browser
to stop (pause at 96 ms of speech; three 32 ms frames arrive in 64 ms). It excludes one
WebSocket hop and the browser's 128-sample render quantum (~3 ms). The same metric on turns
cancelled *before any audio* reads ~195 ms — that is the 250 ms confirmation path, and
nothing was audible to cut.

The agent's next turn answered the interruption itself ("Understood. Let me know if you need
anything else."), with only the heard fragment in its history.

## Tools

| Clip | Tool called | Result |
| --- | --- | --- |
| "What is eighteen times forty two?" | `calculate("18 * 42")` → 756 | Correct |
| "What's fifteen percent of two thousand four hundred?" | none | Model answered 360 itself — correct. The trigger missed "15%" followed by "of" |
| "What time is it right now?" | none (offered) | Model asked for a timezone instead of calling `current_time` |

One of three. The 1.5B model's tool use is the weak point the roadmap predicted.

## Found while building and measuring

- A sentence whose synthesis had begun but sent no audio counted as fully heard. Unit test.
- The browser's playback report had nowhere to land once the turn took ownership of the cut
  record. Unit test.
- `replay.py`: turns now end when their audio has played, so a 60-second reply left the
  harness waiting longer than its timeout, and its idle check waited on a few milliseconds
  of virtual playback. Two runs were truncated and rerun.
- `bench.py` looked up `onnxruntime`, which the GPU build replaces.

## Not yet right

- **p50 is inside budget, p90 is not** (1110 ms). The tail is mostly the first chunk: a long
  opening clause still synthesises whole.
- **Speculation wastes work:** 12 cancelled starts in 32 turns, and first-token latency rose.
  Cancelling the vLLM request sooner, or not starting generation until transcription has
  settled, are the next things to try.
- **Tool calls** (above), and the trigger for "N% of M".
- Barge-in was measured with the virtual player, not a real microphone and speaker; echo
  from laptop speakers can still trigger onsets (`MURMUR_BARGE_IN=false` exists for that).
- The 1.5B model ignores "one to three short sentences". The 96-token cap in these runs is a
  measurement convenience, not a fix.
