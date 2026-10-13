"""The scripted memory check from docs/02, against the real LLM and embedder.

1. Write: 124 scripted turns (dev and held-out facts, restatements, contradictions, small
   talk), extracted and written without consolidation. Count the store.
2. Consolidate: one batch pass. Count again.
3. Restart: reopen the store from disk in a fresh MemoryService.
4. Recall: one question per fact plus contradiction checks. Hit rate, latency, and whether
   the spoken answer uses the fact.

Held-out facts are always written but only scored with --final. Tune on the dev set; run
--final once when tuning is frozen, or the held-out numbers stop meaning anything.

Twenty-odd scripted facts is not a benchmark. It shows the layer works and catches regressions.

    uv run python scripts/memory_eval.py [--final]      # needs `make serve`
"""

import asyncio
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from murmur.agent.loop import Agent
from murmur.agent.memory_jobs import MemoryJobs
from murmur.config import get_settings
from murmur.llm import build_llm
from murmur.memory import open_memory
from murmur.pipeline.metrics import now_ms


@dataclass(frozen=True)
class Fact:
    statement: str
    restatement: str
    question: str
    key: str  # regex the recalled fact (and ideally the answer) must match


FACTS = [
    Fact(
        "My sister is called Afifa.",
        "Afifa, my sister, is visiting this weekend.",
        "What's my sister's name?",
        r"afifa",
    ),
    Fact(
        "I play badminton on weekday evenings.",
        "Badminton after work again tonight, as usual.",
        "What sport do I play?",
        r"badminton",
    ),
    Fact(
        "I work as a backend engineer at a fintech startup.",
        "Long day of backend work at the startup.",
        "What do I do for work?",
        r"backend|engineer|fintech",
    ),
    Fact(
        "My dog is a beagle named Biscuit.",
        "Biscuit the beagle chewed my shoe.",
        "What's my dog called?",
        r"biscuit",
    ),
    Fact(
        "I'm allergic to peanuts.",
        "Remember, no peanuts for me, I'm allergic.",
        "Is there any food I need to avoid?",
        r"peanut",
    ),
    Fact(
        "My favourite food is biryani.",
        "Nothing beats a plate of biryani, my favourite.",
        "What's my favourite dish?",
        r"biryani",
    ),
    Fact(
        "I'm learning to play the guitar.",
        "Practised guitar chords for an hour today.",
        "Which instrument am I learning?",
        r"guitar",
    ),
    Fact(
        "My birthday is on the 3rd of March.",
        "My birthday, the third of March, is coming up.",
        "When is my birthday?",
        r"march",
    ),
    Fact(
        "I drive a red Hyundai i20.",
        "Took my red i20 for a service.",
        "What car do I drive?",
        r"hyundai|i20",
    ),
    Fact(
        "I don't eat meat on Tuesdays.",
        "It's Tuesday, so no meat for me today.",
        "Do I have any diet rules on particular days?",
        r"tuesday",
    ),
    Fact(
        "My best friend is Rahul, we went to college together.",
        "Rahul, my best friend from college, called today.",
        "Who is my best friend?",
        r"rahul",
    ),
    Fact(
        "I usually wake up at 6 in the morning.",
        "Up at six again, my usual time.",
        "What time do I usually get up?",
        r"\b6\b|six",
    ),
    Fact(
        "I'm training for a half marathon in December.",
        "Did a long run for the December half marathon.",
        "What race am I training for?",
        r"marathon",
    ),
    Fact(
        "I prefer tea over coffee.",
        "I'll have tea, I always prefer it to coffee.",
        "Should you offer me coffee or tea?",
        r"tea",
    ),
    Fact(
        "My mother lives in Hyderabad.",
        "Called my mum in Hyderabad this morning.",
        "Where does my mom live?",
        r"hyderabad",
    ),
    Fact(
        "I'm reading Sapiens at the moment.",
        "Got through another chapter of Sapiens.",
        "What book am I reading?",
        r"sapiens",
    ),
    Fact(
        "I really dislike horror movies.",
        "Horror films are just not for me.",
        "Would I enjoy a horror film tonight?",
        r"horror",
    ),
    Fact(
        "I speak Urdu, Hindi and English.",
        "Switched between Urdu and English all day.",
        "Which languages do I speak?",
        r"urdu",
    ),
    Fact(
        "My daughter Zara is five years old.",
        "Zara, my five-year-old, drew me a picture.",
        "How old is my daughter?",
        r"zara|five|\b5\b",
    ),
    Fact(
        "I'm saving up for a trip to Japan next year.",
        "Put more money aside for Japan.",
        "Where am I planning to travel?",
        r"japan",
    ),
]


@dataclass(frozen=True)
class Contradiction:
    before: str
    after: str
    question: str
    new_key: str
    old_key: str


CONTRADICTIONS = [
    Contradiction(
        "I live in Chennai.",
        "I've just moved to Bangalore for good.",
        "Which city do I live in?",
        r"bangalore",
        r"chennai",
    ),
    Contradiction(
        "My phone is an iPhone 12.",
        "I switched to a Pixel 8 last week.",
        "What phone do I use?",
        r"pixel",
        r"iphone",
    ),
    Contradiction(
        "I'm a big Chelsea fan.",
        "I've stopped supporting Chelsea, I support Arsenal now.",
        "Which football team do I support?",
        r"arsenal",
        r"chelsea",
    ),
]

SMALL_TALK = [
    "What's the weather like today?",
    "Tell me a joke.",
    "How do I boil an egg?",
    "Thanks!",
    "What time is it in London?",
    "Can you explain what a black hole is?",
    "Hmm, okay.",
    "How many ounces in a pound?",
    "What's a good name for a goldfish?",
    "Never mind.",
    "Who wrote Pride and Prejudice?",
    "Give me a quick stretching tip.",
    "That's funny.",
    "What's the capital of Australia?",
    "How do you say thank you in French?",
    "Hello?",
    "Can you set a timer for ten minutes?",
    "What's two hundred divided by eight?",
    "Cool.",
    "Recommend a podcast about history.",
    "What does API stand for?",
    "Sorry, go on.",
    "How long should I steep green tea?",
    "Is it going to rain tomorrow?",
    "Okay, bye for now.",
    "Explain compound interest simply.",
    "Why is the sky blue?",
]


# First held-out set. Its results were printed during tuning round 3, so it is no longer
# held out; it now counts as development data.
SEEN_FACTS = [
    Fact("My wife Sana is a dentist.", "", "What does my wife do for a living?", r"dentist"),
    Fact("I grew up in Lucknow.", "", "Where did I grow up?", r"lucknow"),
    Fact("I'm lactose intolerant.", "", "Can I have a milkshake?", r"lactose"),
    Fact("My favourite cricketer is Rohit Sharma.", "", "Who's my favourite cricketer?", r"rohit"),
    Fact("I go swimming every Sunday morning.", "", "What do I do on Sunday mornings?", r"swim"),
    Fact("My cat is called Pistachio.", "", "What's my cat's name?", r"pistachio"),
    Fact("I'm studying for my AWS certification.", "", "Which exam am I preparing for?", r"aws"),
    Fact("I'm scared of flying.", "", "How do I feel about planes?", r"fly|flying|flight"),
]

SEEN_CONTRADICTIONS = [
    Contradiction(
        "I work night shifts at the hospital.",
        "I've switched to day shifts at the hospital now.",
        "What shifts do I work?",
        r"\bday\b",
        r"night",
    ),
    Contradiction(
        "My laptop is a ThinkPad.",
        "I replaced my ThinkPad with a MacBook.",
        "What laptop do I use?",
        r"macbook",
        r"thinkpad",
    ),
]


# Held out: written to the store on every run, scored only with --final, after tuning froze.
# No name, place or topic here appears in any prompt example or in the development facts.
HELDOUT_FACTS = [
    Fact(
        "My uncle Farhan teaches physics at a college.",
        "",
        "What does my uncle do?",
        r"physics|teach",
    ),
    Fact("I collect old coins.", "", "What's my hobby?", r"coin"),
    Fact("I can't stand loud music.", "", "Should you put on some loud music for me?", r"loud"),
    Fact("My go-to breakfast is poha.", "", "What do I usually eat for breakfast?", r"poha"),
    Fact(
        "My nephew Kabir turns ten next month.",
        "",
        "How old will my nephew be?",
        r"kabir|ten|\b10\b",
    ),
    Fact(
        "I'm colourblind.",
        "",
        "Is there anything about my eyesight you should know?",
        r"colou?r ?blind",
    ),
    Fact(
        "I volunteer at an animal shelter on Saturdays.",
        "",
        "What do I do on Saturdays?",
        r"shelter|volunteer",
    ),
    Fact("My favourite author is Ruskin Bond.", "", "Who is my favourite writer?", r"ruskin|bond"),
]

HELDOUT_CONTRADICTIONS = [
    Contradiction(
        "I drink three cups of coffee a day.",
        "I've quit coffee completely.",
        "Do I still drink coffee?",
        r"quit|stopped|no longer|gave up|doesn't|does not",
        r"three cups|3 cups",
    ),
    Contradiction(
        "My team at work writes everything in Java.",
        "We've migrated all our services from Java to Go.",
        "What language does my team use?",
        r"\bgo\b|golang",
        r"java",
    ),
]

DEV_FACTS = FACTS + SEEN_FACTS
DEV_CONTRADICTIONS = CONTRADICTIONS + SEEN_CONTRADICTIONS


def scripted_turns(seed: int = 7) -> list[str]:
    early = [f.statement for f in DEV_FACTS + HELDOUT_FACTS]
    early += [c.before for c in DEV_CONTRADICTIONS + HELDOUT_CONTRADICTIONS]
    late = [f.restatement for f in DEV_FACTS if f.restatement]
    late += [c.after for c in DEV_CONTRADICTIONS + HELDOUT_CONTRADICTIONS]
    chatter = SMALL_TALK * 2
    rng = random.Random(seed)
    rng.shuffle(early)
    rng.shuffle(late)
    rng.shuffle(chatter)
    # Every "after" lands later than its "before"; chatter is spread through both halves.
    half = len(chatter) // 2
    first, second = early + chatter[:half], late + chatter[half:]
    rng.shuffle(first)
    rng.shuffle(second)
    return first + second


def matches(pattern: str, texts: list[str]) -> bool:
    return any(re.search(pattern, t, re.IGNORECASE) for t in texts)


async def answer(llm: Any, question: str, memories: list[str]) -> str:
    agent = Agent(llm, max_turns=1)  # fresh: only long-term memory can supply the fact
    return "".join([d async for d in agent.respond(question, memories)])


async def score_facts(
    llm: Any, memory: Any, k: int, facts: list[Fact], latencies: list[float]
) -> list[dict[str, Any]]:
    rows = []
    for fact in facts:
        start = now_ms()
        recalled = [m.text for m in memory.recall(fact.question, k)]
        latencies.append(now_ms() - start)
        reply = await answer(llm, fact.question, recalled)
        rows.append(
            {
                "question": fact.question,
                "recalled": recalled,
                "hit": matches(fact.key, recalled),
                "answered": matches(fact.key, [reply]),
                "reply": reply,
            }
        )
    return rows


async def score_contradictions(
    llm: Any, memory: Any, k: int, cases: list[Contradiction]
) -> list[dict[str, Any]]:
    rows = []
    every = memory.all(include_superseded=True)
    for c in cases:
        recalled = [m.text for m in memory.recall(c.question, k)]
        reply = await answer(llm, c.question, recalled)
        rows.append(
            {
                "question": c.question,
                "recalled": recalled,
                "new_surfaces": matches(c.new_key, recalled),
                # A recalled fact naming the old value but not the new one, e.g. "Owns an iPhone".
                "stale_surfaces": any(
                    re.search(c.old_key, t, re.I) and not re.search(c.new_key, t, re.I)
                    for t in recalled
                ),
                "old_superseded": any(
                    not m.active
                    and re.search(c.old_key, m.text, re.I)
                    and not re.search(c.new_key, m.text, re.I)
                    for m in every
                ),
                "answer_has_new": matches(c.new_key, [reply]),
                "reply": reply,
            }
        )
    return rows


def summary(rows: list[dict[str, Any]], key: str) -> str:
    return f"{sum(r[key] for r in rows)}/{len(rows)}"


async def main() -> int:
    settings = get_settings()
    out_dir = settings.results_dir / "memory_eval"
    shutil.rmtree(out_dir, ignore_errors=True)
    settings = settings.model_copy(update={"memory_db": out_dir / "memory.db"})
    llm = build_llm(settings)

    # 1. Write without consolidation.
    memory = open_memory(settings)
    jobs = MemoryJobs.from_settings(llm, memory, settings)
    turns = scripted_turns()
    chatter_with_facts: list[dict[str, Any]] = []
    for i, text in enumerate(turns):
        facts = await jobs.extract(text, "")
        memory.add(facts, f"eval_{i}")
        if text in SMALL_TALK and facts:
            chatter_with_facts.append({"turn": text, "facts": facts})
    before = len(memory.all(include_superseded=False))
    print(f"write: {len(turns)} turns -> {before} records (no consolidation)")

    # 2. Consolidate.
    await jobs.consolidate_all()
    after = len(memory.all(include_superseded=False))
    superseded = [m for m in memory.all(include_superseded=True) if not m.active]
    print(f"consolidate: {after} active, {len(superseded)} superseded")

    # 3. Restart: a fresh service over the same file.
    memory = open_memory(settings)
    memory.recall("warm up", k=1)

    # 4. Recall.
    k = settings.recall_k
    latencies: list[float] = []
    final = "--final" in sys.argv
    dev = await score_facts(llm, memory, k, DEV_FACTS, latencies)
    dev_c = await score_contradictions(llm, memory, k, DEV_CONTRADICTIONS)
    heldout = await score_facts(llm, memory, k, HELDOUT_FACTS, latencies) if final else []
    heldout_c = await score_contradictions(llm, memory, k, HELDOUT_CONTRADICTIONS) if final else []

    per_100 = 100 / len(turns)
    report = {
        "turns": len(turns),
        "records_before_consolidation": before,
        "records_after_consolidation": after,
        "records_per_100_turns": {
            "before": round(before * per_100, 1),
            "after": round(after * per_100, 1),
        },
        "superseded": [m.text for m in superseded],
        "small_talk_turns": len(SMALL_TALK) * 2,
        "small_talk_turns_that_wrote_facts": chatter_with_facts,
        "dev": {
            "recall_hit_rate": summary(dev, "hit"),
            "answer_uses_fact": summary(dev, "answered"),
        },
        "heldout": {
            "recall_hit_rate": summary(heldout, "hit"),
            "answer_uses_fact": summary(heldout, "answered"),
        },
        "contradictions": {
            "dev": {
                "new_surfaces": summary(dev_c, "new_surfaces"),
                "stale_surfaces": summary(dev_c, "stale_surfaces"),
                "old_superseded": summary(dev_c, "old_superseded"),
                "answer_has_new": summary(dev_c, "answer_has_new"),
            },
            "heldout": {
                "new_surfaces": summary(heldout_c, "new_surfaces"),
                "stale_surfaces": summary(heldout_c, "stale_surfaces"),
                "old_superseded": summary(heldout_c, "old_superseded"),
                "answer_has_new": summary(heldout_c, "answer_has_new"),
            },
        },
        "mean_memories_returned": round(
            float(np.mean([len(r["recalled"]) for r in dev + heldout])), 2
        ),
        "recall_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 1),
            "p90": round(float(np.percentile(latencies, 90)), 1),
            "max": round(float(np.max(latencies)), 1),
            "n": len(latencies),
        },
        "store_size_at_recall": after,
        "settings": {
            "recall_k": k,
            "recall_min_similarity": settings.recall_min_similarity,
            "duplicate_threshold": settings.memory_duplicate_threshold,
            "related_threshold": settings.memory_related_threshold,
            "embed_model": settings.embed_model,
            "llm_model": settings.llm_model,
        },
        "facts": dev,
        "heldout_facts": heldout,
        "contradiction_cases": dev_c + heldout_c,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    for name in ("records_per_100_turns", "dev", "heldout", "contradictions", "recall_ms"):
        print(f"{name}: {report[name]}")
    print(f"small talk that wrote facts: {len(chatter_with_facts)}/{len(SMALL_TALK) * 2}")
    for r in dev + heldout:
        if not r["hit"]:
            print(f"  MISS  {r['question']}  recalled={r['recalled']}")
    for c in dev_c + heldout_c:
        print(f"  {c['question']}  recalled={c['recalled']}")
    print(f"superseded: {report['superseded']}")
    print(f"wrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
