# Murmur — TODO

Ten work days. Tick as you go. A day that ends without its boxes ticked is a day
to cut scope from, not a day to extend.

---

## Day 0 — Setup (1 hour, tonight)

- [ ] `git init`, push with `CLAUDE.md`, `README.md`, `TODO.md`, `docs/`
- [ ] Turn off AI attribution trailers in your git tooling
- [x] `uv venv`, dependencies, pin versions
- [x] Secret scan pre-commit hook
- [x] `.gitignore`: `models/`, `results/`, `.env`, `*.wav`, `__pycache__/`
- [x] One hour trying Pipecat and LiveKit; pick one and write down why
  (hand-rolled instead — see docs/03 Decisions)

---

## Days 1–3 — Make it talk

### Audio
- [x] Mic capture in the browser, frames to the server
- [x] Playback back to the browser, **cancellable mid-stream**
- [x] Silero VAD wired in, speech/silence events
- [x] Record 10 test audio clips for replay testing
  (stand-in: 10 MInDS-14 clips, `clips/README.md`; own recordings still to do)

### Stages
- [x] `stt/base.py` + streaming local implementation
- [x] `llm/base.py` + vLLM local backend
- [x] `tts/base.py` + streaming local synthesis
- [x] `agent/loop.py` — text in, text out, no audio dependency

### Wiring
- [x] `pipeline/orchestrator.py` — one task group per turn
- [x] **Cancellation designed in from the start**, not retrofitted
- [x] `pipeline/metrics.py` — per-stage timing, the only timing code
- [x] Turn records written to `results/`

### Gate 1
- [x] It holds a spoken conversation
- [x] Time to first audio measured and recorded, however bad
- [x] VRAM fits with all three models loaded
- [x] `reports/R1-pipeline.md` — where the time goes, what you attack first

---

## Days 4–6 — Make it remember

### Store
- [x] SQLite schema + vector index
- [x] Memory record type with `superseded_by`, `use_count`, `last_used_at`

### Operations
- [x] `write` — extract atomic facts after a turn, **async, off the turn path**
- [x] `recall` — hybrid retrieval, **40ms budget, on the turn path**
- [x] `consolidate` — merge, supersede, decay; batched
- [x] `forget` — hard delete, exposed as a tool

### Integration
- [x] Memory MCP server
- [x] Recall wired into prompt assembly inside budget
- [x] Working memory (session) separate from long-term

### Checks
- [x] 20 scripted facts stated, second conversation retrieves them
- [x] Recall precision recorded
- [x] Contradiction test: state, contradict, check which surfaces
- [x] Store growth per 100 turns, before and after consolidation

### Gate 2
- [x] Recalls a fact across a process restart
- [x] Recall stays inside 40ms
- [x] `reports/R2-memory.md`

---

## Days 7–8 — Make it feel real

- [x] Barge-in: stop playback, cancel generation, drop queued synthesis
- [ ] Partial turns recorded as **spoken**, not generated
- [ ] Endpointing tuned; both failure modes measured and written down
- [x] Sentence-level TTS handoff
- [x] Tools MCP server (keep it small — 2 or 3 tools)
- [ ] Optimisation pass against the budget table

### Gate 3
- [ ] Barge-in cuts audio within 200ms
- [ ] Time to first audio inside 800ms at p50
- [ ] `reports/R3-latency.md` — before and after, per stage

---

## Days 9–10 — Ship it

- [ ] Hosted backend behind `MURMUR_BACKEND=hosted`
- [ ] Deploy; public demo link works
- [ ] Demo page: talk, see the transcript, see what it remembered
- [ ] Local vs hosted latency comparison published
- [ ] README numbers generated from `results/latency.json` by script
- [ ] Recorded walkthrough for when the demo is down
- [ ] `reports/R4-final.md`
- [ ] Resume bullets written from real numbers

---

## Stretch — only if days remain

- [ ] Fine-tune the local model for tool calling against your MCP schemas
- [ ] Report tool-call validity before and after

Do not start this before day nine.

---

## If you fall behind

Cut in this order: hosted backend, demo page polish, tools server, consolidation.

**Never cut:** latency instrumentation, memory recall, barge-in.

---

## Not in this project

- Multi-user accounts
- Telephony
- Wake word
- Speaker diarisation
- Multi-language
- Kubernetes
