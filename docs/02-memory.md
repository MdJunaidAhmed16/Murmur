# 02 — Memory

The technical core. Everyone dumps chat history into the context window and calls
it memory; this is the part that makes the project worth discussing for forty
minutes.

## The problem with transcripts

Replaying the conversation into context fails in three ways. It grows without
bound, so it eventually exceeds the window and costs latency long before that. It
has no notion of importance, so "my sister is called Afifa" and "um, hold on" get
equal weight. And it cannot handle change — if you said you lived in Chennai last
month and Bangalore today, a transcript contains both and the model picks whichever
it happens to attend to.

Memory is a different data structure, not a longer transcript.

## Two tiers

**Working memory** is the current session: the full exchange, in order, in
context. Bounded by a turn count. Nothing clever.

**Long-term memory** is what survives the session: atomic facts, stored
deliberately, retrieved selectively.

The interesting engineering is entirely in the second.

## A memory record

```json
{
  "id": "mem_01H...",
  "text": "Plays badminton on weekday evenings",
  "created_at": "2026-10-14T19:02:11Z",
  "last_used_at": "2026-10-16T08:30:02Z",
  "use_count": 3,
  "source_turn": "turn_...",
  "superseded_by": null,
  "embedding": [...]
}
```

Facts are atomic and self-contained. "Plays badminton on weekday evenings" is one
memory. "Plays badminton and works at a software company and has a sister" is
three, and storing it as one makes retrieval worse at every future turn.

## Three operations

### Write

After a turn ends — **off the critical path** — decide whether anything durable was
said, and if so extract it as atomic facts.

Most turns produce nothing. "What time is it" is not a memory. Be aggressive about
storing nothing; a store full of noise retrieves noise.

The extraction prompt asks for facts that would still be true next week. That
single framing does most of the filtering work.

### Consolidate

Also off the critical path, batched, runs after the session or on a timer.

Three jobs. **Merge** near-duplicates into one record. **Supersede** contradicted
facts — the old record is marked rather than deleted, so you keep an audit trail
and can explain where a belief came from. **Decay** by marking facts that have not
been retrieved in a long time as low priority, so retrieval stops surfacing them
without losing them.

Consolidation is the slowest operation in the system and the user must never wait
for it.

### Recall

On the critical path, inside the 250ms model budget. This is the one that has to
be fast.

Hybrid retrieval over the fact store: vector similarity for semantic match, plus a
recency and use-count boost so recently useful facts surface first. Return a small
number — three to five — because stuffing twenty memories into the prompt costs
prefill time and degrades the answer.

Budget it explicitly: recall gets 40ms. If it exceeds that, cut `k` before you cut
anything else.

## The critical-path rule

```
TURN:  recall (40ms, blocking)  →  generate  →  speak
AFTER: extract and write (async)
LATER: consolidate (batched)
```

Write this rule into `memory/__init__.py` as a comment. It is the design decision
that most easily erodes, because adding "just one quick write" to the turn path is
always tempting and always wrong.

## Contradiction

When a new fact contradicts a stored one, the new fact wins and the old one gets
`superseded_by` pointing at it. Never delete.

This gives you something genuinely nice for the demo: ask it "what did you think
before?" and it can answer, because the history is intact.

## Forgetting

The user must be able to say "forget that" and have it work. Implement it as a
tool the agent can call, and make it a hard delete rather than a supersede —
when someone asks you to forget something, keeping it is the wrong answer.

## Measuring it

Memory quality is harder to measure than latency, and the honest version is small
and manual:

- A fixed set of 20 facts stated across a scripted conversation.
- A second scripted conversation that should retrieve each one.
- **Recall precision**: of the facts that should have surfaced, how many did.
- **Store growth**: records per 100 turns, before and after consolidation.
- **Contradiction handling**: state a fact, contradict it, check which one surfaces.

Twenty scripted facts is not a benchmark, and the README should say so. It is
enough to show the layer works and to catch regressions.
