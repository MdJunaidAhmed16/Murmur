# R2 — Memory

Gate 2: recalls a fact across a process restart, recall stays inside 40 ms, and the
scripted check passes with recall precision recorded.

> Eval figures are from `results/memory_eval/report.json` (`scripts/memory_eval.py`);
> earlier rounds are kept as `results/memory_eval-v1..v5.json`. Live figures are from
> `results/turns.jsonl`. Twenty-odd scripted facts is not a benchmark — it shows the
> layer works and catches regressions.

## Gate 2

| Check | Result |
| --- | --- |
| Recall across a process restart | **Pass.** Fact spoken to process 1, recalled by a fresh process 2 through the full voice pipeline |
| Recall inside 40 ms | **Pass.** 8–10 ms in live turns. Eval: p50 16.6 ms, max 33.3 ms (n=36) |
| Scripted check, recall precision recorded | **Recorded:** dev 24/28, held-out 4/8 (below) |
| Forget | **Pass.** Spoken "please forget … about my sister" → `forget` tool over MCP → hard delete → next question gets "I don't have any information about your sister" |

Recall in the eval is slower than in live turns because the eval calls it straight after
a generation, while vLLM's process is still busy on the CPU. The 40 ms budget is enforced
regardless: over budget, the turn proceeds without memories.

## Design as built

```
TURN:  recall (embed query ~6 ms + matrix scoring, budget 40 ms)  →  generate  →  speak
AFTER: one worker — extract → ground → write → consolidate (waits for the voice loop to idle)
```

- **Store.** SQLite (WAL) holds records and embeddings; recall reads an immutable
  in-memory snapshot swapped on each write, so it never touches the database or a lock.
- **Embeddings.** bge-small-en-v1.5 via fastembed (ONNX, CPU, 384-d). LlamaIndex was
  dropped: for one user's few hundred facts, a numpy dot product is sub-millisecond and
  the framework would only add latency and dependencies.
- **Ranking.** cosine + small recency (30-day half-life) and use-count boosts − a decay
  penalty. Floor 0.5, top 4.
- **Extraction.** few-shot prompt → reject placeholders → **grounding check** (a fact must
  share a content word with what the user said) → write. Turns with no first-person words,
  not answering a question about the user, are skipped without calling the model.
- **Consolidation.** near-identical text (≥ 0.98) merges. Otherwise, older facts with
  similarity ≥ **0.70** go to an LLM judge **one pair at a time** (same / replaced / keep).
  Replaced → `superseded_by`, kept for history. Decay marks unused facts low-priority.
- **Forget.** A tool on the memory MCP server (in-process `Client`), offered to the model
  only when the user says forget/delete/erase. It drains pending writes first, hard-deletes
  matches and the history they superseded, and the agent scrubs the session transcript.
- **Recall is not an MCP call.** It sits too close to the budget to pay for a protocol hop.

## How it got here: five eval rounds

Same 100 scripted turns (later 124, adding held-out facts), Qwen2.5-1.5B doing extraction
and judging.

| Round | Change | Records / 100 turns (before → after consolidation) | Small talk that wrote facts | Dev recall |
| --- | --- | --- | --- | --- |
| v1 | baseline | 280 → 116 | 18 / 54 | 12 / 20 |
| v2 | few-shot prompt, grounding check, recall floor 0.6 → 0.5, merge only ≥ 0.98 | 77 → 52 | 30 / 54 | 20 / 20 |
| v3 | skip turns not about the user | 49 → 39 | 2 / 54 | 19 / 20 |
| v4 | judge one pair at a time | 50 → 25 | 0 / 54 | 20 / 28 |
| v5 | judge only pairs with similarity ≥ 0.70 | 50 → 35.5 | 0 / 54 | 24 / 28 |

What each round found:

- **v1: the prompt contaminated the eval.** Its only example was "Has a sister called
  Afifa" — one of the test facts. The model copied it onto unrelated turns and invented
  generic facts ("lives in New York City", "engineer at Google"). The grounding check and
  example-free-of-eval prompt fixed most of it.
- **v1: embedding-based merging is unsafe.** "Prefers tea over coffee" merged into
  "prefers coffee over tea" at 0.92 — embeddings barely see word order or negation. Only
  near-identical text (≥ 0.98) now merges without the judge.
- **v2: no similarity threshold separates right from wrong** while the store is noisy:
  correct question→fact matches scored 0.53–0.76, wrong ones 0.45–0.69. Cleaning the store
  mattered more than tuning the floor.
- **v3 → v4: the judge.** Judging a list at once, it superseded too little (1 of 3); judging
  pairs, far too much (23 supersedes: cat replaced dog, daughter replaced wife). A wrong
  supersede silently loses a true fact; a missed one only lets a stale fact surface next to
  the new one. So precision matters more.
- **v5: an embedding gate before the judge.** True contradictions scored 0.64–0.997, the
  judge's false supersedes 0.61–0.71. Judging only pairs ≥ 0.70 kept 6 of 7 true pairs and
  blocked 11 of 12 false ones.

Tuning stopped at v5. Dev answers moved 24 → 22 between identical runs (answers are sampled
at temperature 0.7) — that is the run-to-run noise on these numbers.

## Final scripted check

124 turns: 28 dev facts (20 stated twice), 5 dev and 2 held-out contradictions, 8 held-out
facts, 54 small-talk turns. Store reopened from disk before recall.

| | Dev (tuned on) | **Held-out (never tuned on)** |
| --- | --- | --- |
| Recall precision (expected fact in top 4) | 24 / 28 | **4 / 8** |
| Spoken answer uses the fact | 22 / 28 | **5 / 8** |

"Recall precision" follows `docs/02`: of the facts that should have surfaced, how many did.
Mean memories returned per query: 3.75.

The held-out gap is real and is why the set exists. Its four misses:

- **Loud music** — a scorer artefact: the right fact was recalled, paraphrased as "Prefers
  quiet music", and the scorer wanted the word "loud". Left as a miss rather than changing
  the scorer after seeing results.
- **Breakfast (poha)** — lost to a false supersede ("Prefers biryani" replaced it).
- **Hobby (coins), colour blindness** — never extracted. Extraction recall is the 1.5B
  model's limit.

**Store growth:** 50 records per 100 turns before consolidation, 35.5 after (10 superseded,
the rest merged).

## Contradictions

| | Dev (5) | Held-out (2) |
| --- | --- | --- |
| New value surfaces | 5 / 5 | 2 / 2 |
| Old record superseded | 4 / 5 | 2 / 2 |
| Stale value still surfaces | 2 / 5 | 0 / 2 |
| Spoken answer gives new value | 4 / 5 | 1 / 2 |

- **Phone (iPhone → Pixel)** was not superseded: similarity 0.64, just under the gate —
  the one true pair the 0.70 gate gives up.
- **Laptop** counts as "stale surfaces" because "Used to have a ThinkPad" was recalled —
  a correct past-tense fact, so a scoring nuance rather than an error.
- **City** passes only partly by design: extraction turned "I've just moved to Bangalore
  for good" into "Works in Bangalore" and a wrong "Lived in Hyderabad before". The old
  "Lives in Chennai" was superseded, but by the wrong fact.
- False supersedes remain among short "Prefers X" facts (biryani, poha, guitar).

## Found in the live pipeline

- **Forgetting did not reach working memory.** After `forget`, the agent still answered
  from the session transcript. The agent now scrubs history messages that share content
  with the forgotten facts.
- **The model called `forget` on a plain statement** ("I play badminton…"), which also
  skipped writing it. Fixed by offering the tool only on forget-intent words — this also
  keeps the schema out of most prompts.
- **Early endpoints split requests.** "Please forget what I told you, … about my sister"
  became two turns, and the second had no "forget" in it. Fixed by **turn continuation**:
  a turn cut off before the agent said anything is merged into the next one. A premature
  endpoint now costs wasted compute, not a broken turn. (Endpoint tuning is still day 7.)
- **A fire-and-forget task could be garbage-collected** before sending its event (asyncio
  holds tasks weakly). Now kept in a set until done.
- **Proper nouns.** Whisper heard "Afifa" as "a Phiffa", and that is what was stored.

## Not yet right

- Extraction recall and contradiction judging are limited by the 1.5B model. The hosted
  backend (day 9) can take the background jobs without touching the voice path.
- "What did you think before?" — superseded history is kept and exposed by the MCP
  server's `history` tool, but the agent is not offered it yet.
- Whisper could be primed with names from memory (`initial_prompt`) to fix proper nouns.
- Consolidation compares each new fact with older ones only; facts written in the same
  turn are not checked against each other.
