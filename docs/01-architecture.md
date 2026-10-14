# 01 — Architecture

## The pipeline

```
   mic
    │
    ▼
┌─────────┐   speech/silence
│   VAD   │──────────────────┐
└────┬────┘                  │
     │ audio frames          │ endpoint signal
     ▼                       ▼
┌──────────────┐      ┌─────────────┐
│ streaming    │─────▶│ turn        │
│ STT          │      │ detector    │
└──────────────┘      └──────┬──────┘
   partials                  │ final transcript
                             ▼
                    ┌─────────────────┐      ┌──────────────┐
                    │   agent loop    │◀────▶│ MCP servers  │
                    │  model + tools  │      │  memory      │
                    └────────┬────────┘      │  tools       │
                             │ token stream  └──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ streaming TTS   │──▶ speaker
                    └─────────────────┘
                             ▲
                    barge-in cancels here
```

Everything streams. Nothing waits for a previous stage to finish before starting,
because waiting is what blows the budget.

## The latency budget

This table is the project's contract. Every design decision is judged against it.

| Stage | Budget | What it covers |
| --- | --- | --- |
| Endpointing | 200 ms | Silence after speech before the turn is declared over |
| Transcription (final) | 100 ms | Finalising the last partial; earlier audio is already done |
| Model, first token | 250 ms | Prompt assembly, memory retrieval, prefill, first token |
| Speech, first audio | 200 ms | Synthesising the first phrase |
| Slack | 50 ms | Buffering and transport |
| **Total to first audio** | **800 ms** | |

**Time to first audio is the headline metric.** Not total response length — the
user hears the first syllable and the conversation feels alive. A three-second
answer that starts in 600ms feels fast; a one-second answer that starts after two
seconds feels broken.

The two biggest levers, in order:

**Endpointing.** The largest single cost and the one with a real trade-off. Too
aggressive and you cut people off mid-thought; too patient and every turn feels
sluggish. Make it configurable, measure both failure modes, and write down where
you landed and why.

**Sentence-level TTS handoff.** Do not wait for the model to finish. Buffer tokens
until the first sentence boundary, send that to synthesis, and keep going. This
single decision is worth more than any model optimisation.

## Barge-in

When the user starts speaking while the agent is talking, three things must
happen immediately: stop audio playback, cancel the in-flight generation, and
discard any queued synthesis.

The hard part is the partial turn. The agent said half a sentence and the user
interrupted — what the agent actually said, not what it intended to say, is what
goes into the conversation history. Record spoken text, not generated text.

**As built (day 7).** A turn lives until its audio has finished playing, so a reply
still audible is a reply that can be cut. Detection has two stages: 96 ms of speech
*pauses* playback, 250 ms confirms and *clears* it, and a false start *resumes*. On
clear the browser reports how many samples it dropped; a per-sentence ledger of
audio sent maps that back to the words actually heard, and only those are
committed. Details and measurements in `reports/R3-latency.md`.

**Turn-taking.** The turn starts at 200 ms of silence but its first audio is held
until 600 ms (speculative endpointing). A user who carries on in between cancels
the turn before anything is heard, and their words merge into the next turn.

## Modules

```
src/murmur/
├── audio/
│   ├── frames.py         # wire format, PCM conversions
│   └── vad.py            # speech/silence detection, onset and endpoint
│                         # (capture and cancellable playback are AudioWorklets in web/static)
├── stt/
│   ├── base.py           # the interface
│   └── local.py          # streaming transcription
├── tts/
│   ├── base.py           # the interface
│   └── local.py          # streaming synthesis
├── llm/
│   ├── base.py           # ONE interface, both backends
│   ├── local.py          # vLLM
│   └── hosted.py         # API
├── agent/
│   ├── loop.py           # turn handling, tool calls
│   ├── prompts.py        # system prompt and memory prompts
│   ├── memory_jobs.py    # extraction and consolidation, off the turn path
│   └── tools.py          # MCP tool client, intent-gated
├── memory/               # see 02-memory.md
├── mcp_servers/
│   ├── memory.py
│   └── tools.py
├── pipeline/
│   ├── orchestrator.py   # wires the stages, owns cancellation
│   ├── playback.py       # what was heard: audio sent per sentence → words
│   └── metrics.py        # per-stage timing, the only timing code
├── web/                  # demo page
└── config.py             # one Settings object
```

## Boundaries

- **`pipeline/metrics.py` is the only place timing is recorded.** Scattered timing
  code produces numbers that disagree with each other.
- **`llm/base.py` is one interface with two implementations.** No branching on
  backend anywhere else. If you find `if backend == "local"` outside `llm/`, that
  is the bug.
- **STT, TTS and LLM all have a `base.py` interface** for the same reason: you will
  swap at least one of them during the ten days.
- **The agent loop never touches audio.** It receives text and emits text. That
  separation is what makes it testable without a microphone.
- **`memory/` never calls the model directly.** It is given text and returns text.

## Cancellation

Every stage must be cancellable mid-flight, because barge-in can arrive at any
moment. Use structured concurrency — one task group per turn, cancelled as a unit.
A stage that cannot be interrupted will leak audio over the user's next sentence,
and you will spend an evening finding out why.

Design this on day one. Retrofitting cancellation into a pipeline that assumed it
would run to completion is a rewrite.

## Instrumentation

Every turn emits one record:

```json
{
  "turn_id": "uuid",
  "ms": {
    "endpoint": 180, "stt_final": 90, "memory_recall": 40,
    "llm_first_token": 220, "llm_first_sentence": "…", "tts_first_audio": 160,
    "time_to_first_audio": 690
  },
  "barge_in": false,
  "tool_calls": ["recall"],
  "memories_retrieved": 3,
  "backend": "local",
  "spoken_chars": 142
}
```

This exists before any optimisation. It is what `make bench` aggregates and what
the README table reports.

`llm_first_sentence` was added after the first replay run: it is the time from the
first token to the first sentence boundary, when TTS can start. The budget table
has no line for it, but it is real time on the path to first audio (hundreds of ms
when the model opens with a long sentence) and has to come out of the model's
250 ms or the slack.
