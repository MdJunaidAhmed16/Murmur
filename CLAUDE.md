# Murmur — repository guide

A voice agent that runs on local hardware and remembers you across
conversations. You talk, it answers out loud fast enough to feel like a
conversation, it can act through tools mid-sentence, and it still knows you
next week.

Rename the project if you want something better; `murmur` is a placeholder.

Ten days of mornings and evenings.

## Doc map

| File | What it settles |
| --- | --- |
| `README.md` | The public face. Skeleton to fill as numbers land |
| `TODO.md` | The ten-day task list, checkbox by checkbox |
| `docs/01-architecture.md` | The pipeline, the latency budget, module boundaries |
| `docs/02-memory.md` | The memory layer — the technical core |
| `docs/03-tech-stack.md` | What you use, why, and what you'll actually get better at |
| `docs/04-roadmap.md` | Ten days with gates that can fail |
| `docs/05-conventions.md` | Commits, code ethics, comments, testing |

## Non-negotiables

**1. Time to first audio is the metric.** Not total response time. The user hears
something within the budget or the conversation feels broken. Every design
decision is judged against `docs/01-architecture.md`'s budget table.

**2. Memory writes never block the voice loop.** Consolidation is slow and
happens off the critical path. A memory operation that adds latency to a spoken
turn is a bug, not a trade-off.

**3. The model backend is swappable from day one.** Local by default, hosted
behind a flag, one interface. Local keeps iteration free. Hosted keeps the public
demo alive when the laptop is shut.

**4. Instrument before optimising.** The per-stage latency breakdown exists before
any tuning. You cannot fix what you did not measure, and "it feels faster" is not
a number.

**5. No fabricated numbers.** Placeholders are `TBD`, never a plausible figure.

**6. Ship something that talks by day three.** Bad latency, no memory, ugly page —
but it talks. Everything after that is improvement on a working system.

## Commit attribution

Commits carry **no AI attribution trailers.** No `Co-Authored-By` naming an
assistant, no "generated with" footer, no tool name in the body. Turn it off in
your tooling before the first commit. `docs/05-conventions.md` has the format.

## Scope boundary

**In.** English first. Single user. One session at a time. Local and hosted model
backends. Memory that survives restarts. A web demo page.

**Out.** Multi-user accounts. Telephony. Mobile apps. Speaker diarisation.
Multi-language — add it after the English path works, not before.
