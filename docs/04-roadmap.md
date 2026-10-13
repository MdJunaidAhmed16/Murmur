# 04 — Ten days

Days are work units, not calendar days. A weekend holds two or three. Mornings
06:00–09:00, evenings after 21:00.

Order is chosen so you have something that talks by day three, and every stage
after that improves a working system rather than building toward one.

## Days 1–3 — Make it talk

**Build.** Audio capture and playback in the browser. VAD. Streaming STT. The LLM
interface with the local backend. Streaming TTS. The orchestrator wiring them
together. Per-stage timing from the first commit.

No memory. No tools. No barge-in. Just: you speak, it answers out loud.

**Gate.** It holds a conversation. Time to first audio is measured and recorded,
**even if it is terrible.** A 3000ms number you can see is worth more than a fast
system you cannot explain.

**Report.** `reports/R1-pipeline.md` — the latency breakdown per stage, where the
time actually goes, and which stage you will attack first.

If you stop here you have a working local voice assistant with measured latency.
That is already a real project.

## Days 4–6 — Make it remember

**Build.** The memory store. Write, consolidate and recall. The memory MCP server.
Recall wired into the turn path inside its budget. The forget tool.

**Gate.** It recalls a fact across a process restart. Recall stays inside 40ms.
The twenty-fact scripted check passes with recall precision recorded.

**Report.** `reports/R2-memory.md` — recall precision, store growth per 100 turns,
consolidation behaviour, and what the contradiction test did.

This is the stage that makes the project distinctive. Do not rush it to get to
polish.

## Days 7–8 — Make it feel real

**Build.** Barge-in with full cancellation. Endpointing tuned, with both failure
modes measured. Sentence-level TTS handoff. The tools MCP server. Latency
optimisation against the budget table.

**Gate.** Barge-in cuts audio within 200ms and the partial turn is recorded as
spoken, not as generated. Time to first audio is inside 800ms at p50.

**Report.** `reports/R3-latency.md` — before and after the optimisation pass, per
stage, with what each change bought.

## Days 9–10 — Ship it

**Build.** The hosted backend behind the flag. Deploy. The demo page. The README
with real numbers. A recorded walkthrough for when the demo is down.

**Gate.** A stranger can click the link and have a conversation. Every number in
the README comes from `results/latency.json`. The local-versus-hosted latency
comparison is published.

**Report.** `reports/R4-final.md`.

## Stretch, only if days remain

Fine-tune the local model for reliable tool calling against your own MCP schemas.
Report tool-call validity before and after.

Do not plan this into the ten days. If you start training on day five you will
ship nothing.

## If you fall behind

Cut in this order and no other: the hosted backend, the demo page polish, the
tools server, consolidation.

**Never cut:** the latency instrumentation, memory recall, barge-in. Those three
are the project.

## Honest risks

**Audio plumbing eats day one.** It always does. Expect it rather than panicking
at 11pm on day one.

**The 1.5B model may be bad at tool calling.** If it is, cut tools to one or two
simple ones and note the limitation. Do not let it block memory, which matters
more.

**TTS quality may disappoint.** Fast local synthesis sounds synthetic. Decide early
whether you care; for a portfolio demo, fast and robotic beats slow and smooth.
