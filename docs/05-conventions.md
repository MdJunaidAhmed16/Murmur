# 05 — Conventions

## Commits

Conventional Commits with a scope from the module list.

```
<type>(<scope>): <imperative summary, ≤ 72 chars>

<body: what changed and why, wrapped at 80>

<footer: Refs #12 / BREAKING CHANGE: ...>
```

**No AI attribution.** No `Co-Authored-By` naming an assistant, no "generated
with" footer, no tool name anywhere in the message or the PR description. The
commit log records changes to this repository, not how they were typed. Turn this
off in your tooling before the first commit — rewriting history later is worse than
configuring it now.

**Types.** `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`,
`chore`, `bench`.

**Scopes.** `audio`, `stt`, `tts`, `llm`, `agent`, `memory`, `mcp`, `pipeline`,
`web`, `docs`.

```
feat(memory): add supersede on contradicting facts

A new fact that contradicts a stored one sets superseded_by on the old
record rather than deleting it. Keeps the audit trail and lets the agent
answer "what did you think before".

---

perf(tts): hand off at sentence boundaries instead of on completion

Time to first audio drops from 1240ms to 690ms at p50. The model still
generates the full response; synthesis just starts on the first sentence.

---

fix(pipeline): cancel queued synthesis on barge-in

Playback stopped but queued audio frames still drained, so the agent
talked over the user's next sentence. Cancellation now covers the whole
task group.

---

bench(latency): per-stage breakdown, local backend, 50 turns
```

`bench` is a custom type for commits that add or change measured numbers. It makes
the history of your latency claims auditable.

**Rules.** One logical change per commit. No `wip` or `misc` on `main`. Any commit
that changes a published number includes before and after in the body.

## Branching

`main` always runs. Short-lived branches named `<type>/<scope>-<slug>`. Squash
merge. No long-running branches.

## Code ethics

### One implementation per concept

Before writing a function, search for it. The same logic in two places will
diverge, and the copy you did not update is the one that will be running when it
matters.

**Three strikes.** Writing something once is fine. The second time you need it,
extract it. Do not pre-extract on the first write — speculative abstraction is as
expensive as duplication and harder to undo.

**Shared logic lives in the lowest common module.** If `stt/` and `tts/` both need
frame resampling, it belongs in `audio/`, not copied into both.

**Interfaces exist so there is no branching.** `llm/base.py` has two
implementations precisely so that no other file contains
`if backend == "local"`. If such a branch appears outside `llm/`, the interface
is wrong — fix the interface, do not add the branch.

### Comments

Comment the **why**, never the **what**. The code already says what it does.

```python
# Good — explains a decision the code cannot
# 40ms ceiling: recall sits inside the 250ms model budget (see docs/01).
RECALL_TIMEOUT_MS = 40

# Good — warns about a non-obvious consequence
# Record what was spoken, not what was generated; barge-in truncates mid-sentence.
history.append(spoken_text)

# Bad — restates the code
# Set the recall timeout to 40
RECALL_TIMEOUT_MS = 40
```

**Short, at the point of confusion.** One line where someone will pause. Not a
paragraph above the function.

**No essay docstrings.** A module gets one sentence. A function gets one line, and
only if the name is not enough. If a function needs three paragraphs to explain,
the function is wrong — give it a better name, or split it until each piece
explains itself.

**A comment that restates the code gets deleted**, not updated. It is worse than
nothing, because it will go stale and then it lies.

**Comment the surprising, not the routine.** A tuned constant, a workaround for
library behaviour, an ordering that matters, a trade-off you made. Those earn a
line. A for-loop does not.

### Functions

Small, named for what they return or do. If you cannot name it without "and", it
does two things and should be two functions.

No flag parameters that change what a function fundamentally does — that is two
functions wearing one name.

No side effects in anything named like a query. `get_memories` does not write.

### Style

- Python 3.11+, `ruff` for lint and format, `mypy` strict on `src/`.
- Type hints everywhere. Dataclasses or Pydantic for anything crossing a module
  boundary.
- Line length 100.
- No bare `except`. No mutable default arguments. No module-level side effects.
- One `Settings` object in `config.py`, from environment. No scattered
  `os.environ`.
- `structlog`, JSON output. No `print` in `src/`.
- Async throughout the pipeline. One blocking call in the turn path ruins the
  budget.

### Naming

Names in code match names in the docs. Time to first audio is
`time_to_first_audio` everywhere — never `ttfa` in one file and `first_audio_ms`
in another. A reader moving between docs and code should never have to translate.

## Testing

| Layer | Covers | Speed |
| --- | --- | --- |
| Unit | Memory write/consolidate/recall, VAD thresholds, prompt assembly | Fast |
| Contract | MCP tool schemas; the STT/TTS/LLM interfaces | Fast |
| Integration | Full turn with recorded audio in and audio out | Medium |
| Latency | `make bench` over scripted turns, asserts the budget | Slow |

The agent loop takes text and returns text, so it is testable without a
microphone. Keep it that way — a pipeline you can only test by speaking into it is
a pipeline you will stop testing.

Record a handful of real audio clips early and replay them. Manual testing by
talking is slow, inconsistent, and impossible at 11pm with people asleep.

## CI

Every PR: lint, type check, unit and contract tests, secret scan.

Nightly: integration test on recorded audio.

CI fails if time to first audio at p50 exceeds the budget on the recorded set.

## Documentation discipline

- A PR that changes behaviour updates its doc in the same PR.
- A decision made in conversation gets written into a doc, or it does not exist.
- README numbers come from `results/latency.json` by script, never typed.
