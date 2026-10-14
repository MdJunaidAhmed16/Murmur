# R3 — Latency

Gate 3: barge-in cuts audio within 200 ms, the partial turn is recorded as spoken, and time
to first audio is inside 800 ms at p50.

> Figures come from `results/r3/*/latency.json` (`bash scripts/gate3.sh`, then
> `uv run python scripts/r3_table.py`). `TBD` means not yet measured.

## Gate 3

| Check | Result |
| --- | --- |
| Barge-in cuts audio within 200 ms | TBD |
| Partial turn recorded as spoken, not generated | TBD (unit-tested; live run pending) |
| Time to first audio ≤ 800 ms at p50 | TBD |

## Before and after, per stage

One change at a time, same 10 MInDS-14 clips (`clips/`), local backend. p50 / p90 ms.

| Stage | baseline (≈ R1) | + clause first chunk | + GPU TTS | + commit hold (final) |
| --- | --- | --- | --- | --- |
| Endpointing | TBD | TBD | TBD | TBD |
| Transcription (final) | TBD | TBD | TBD | TBD |
| Memory recall | TBD | TBD | TBD | TBD |
| Model, first token | TBD | TBD | TBD | TBD |
| Model, first chunk | TBD | TBD | TBD | TBD |
| Speech, first audio | TBD | TBD | TBD | TBD |
| **Time to first audio** | TBD | TBD | TBD | TBD |

VRAM with all three models resident: TBD.

## What each change is for

**Clause-level first chunk.** Kokoro synthesises a chunk whole, so first audio waits for the
whole first chunk. The first chunk may now end at a comma once it is 24+ characters; later
chunks stay whole sentences because they synthesise while earlier audio plays.

**GPU TTS.** R1 found Kokoro on CPU at 1868 ms p50 to first audio — 72% of the total.
`onnxruntime-gpu` is pinned to 1.26.0, the last CUDA 12 build: later builds ship CUDA 13 with
a `libcudnn.so.9` of the same name as the CUDA 12 one ctranslate2 (Whisper) loads, and two
libraries with one name in one process is a debugging session waiting to happen. One set of
CUDA 12 wheels now serves STT and TTS.

**Speculative endpointing.** See below.

## Endpointing: both failure modes

Measured with the VAD alone (`scripts/endpoint_sweep.py`), 16 single-utterance clips: 10
real callers from MInDS-14 and 6 synthetic.

| Silence threshold | Utterances cut mid-way | Cost added to every turn |
| --- | --- | --- |
| 200 ms | 7 / 16 | 200 ms |
| 300 ms | 6 / 16 | 300 ms |
| 400 ms | 4 / 16 | 400 ms |
| 500 ms | 2 / 16 | 500 ms |
| **600 ms** | **0 / 16** | 600 ms |

Longest pause inside an utterance: 544 ms (a caller's "um…"). Too aggressive cuts people off;
too patient adds the whole threshold to every turn. Neither end is acceptable inside an
800 ms budget.

**Where it landed: speculative endpointing.** The turn *starts* at 200 ms of silence —
transcription, recall and generation run — but its first audio is *held* until 600 ms of
silence. If the user carries on in between, the turn is cancelled before a word is heard
and turn continuation merges what they said into the next turn. The hold costs nothing when
the pipeline is slower than 600 ms anyway, and the wait is inside `time_to_first_audio`, so
it is never hidden. Whether it binds once TTS is on the GPU: TBD.

## Barge-in

Two stages, because the confirmation threshold (250 ms of speech) alone can never meet a
200 ms cut:

1. **Onset** (96 ms of speech while the agent is audible) → `pause`. Playback stops, the
   queue is kept.
2. **Confirmed** (250 ms) → cancel generation and synthesis, `clear` the queue. **Abort**
   (the onset never became speech: a cough, a click, echo) → `resume`.

`barge_in` is measured from the interruption's first speech frame to the server telling the
browser to stop. It excludes one WebSocket hop and one 128-sample render quantum (~3 ms) in
the browser.

A turn now stays alive until its audio has finished playing (by estimate), so cutting in
after generation ends is still a barge-in.

**Recorded as spoken.** On `clear`, the browser reports how many samples it dropped. The
server keeps a ledger of audio sent per sentence and maps the playback position back to
words: whole sentences played, plus the played fraction of the one cut off, rounded down.
Without a report (a non-browser client) it estimates from wall-clock time since first audio.

Barge-in, p50 / p90: TBD.

## Tools

`murmur-tools` MCP server: `current_time` and `calculate` (arithmetic by walking the AST;
never `eval`). Offered to the model only when the user's words call for them, like `forget`
(R2). Live tool-call results: TBD.

## Found while building

- A sentence whose synthesis had begun but sent no audio counted as fully heard (a
  zero-length ledger segment satisfied "played past its end"). Caught by a unit test.
- The browser's playback report arrived after the turn had taken ownership of the cut
  record, so it had nowhere to land. The report future now has its own reference.

## Not yet right

TBD after the run.
