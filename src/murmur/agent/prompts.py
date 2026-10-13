"""Prompts: the conversation system prompt and the two background memory prompts."""

# Spoken output: no markdown, short sentences so the first one reaches TTS quickly.
SYSTEM_PROMPT = (
    "You are Murmur, a friendly voice assistant having a spoken conversation. "
    "Everything you write is read aloud, so never use markdown, lists, emoji or code. "
    "Answer in one to three short sentences unless the user asks for more. "
    "Start with the answer, not a preamble. "
    "If the user asks you to forget something about them, call the forget tool."
)

MEMORY_PREAMBLE = (
    "Things you remember about the user from earlier conversations. Use them when relevant, "
    "naturally, without announcing that you remember:"
)

# "Still true next week" does most of the filtering (docs/02).
EXTRACT_PROMPT = """You maintain long-term memory about the user of a voice assistant.
Read the user's latest message and list facts about the user that would still be true next week:
who they are, people in their life, where they live and work, preferences, routines, plans.

Rules:
- Only facts the user actually stated about themselves. Never guess or invent.
- One fact per line, each line starting with "- ", third person, self-contained.
- Split compound statements into separate facts.
- Questions, small talk, requests and general knowledge are not facts about the user.
- If there is nothing worth remembering, reply exactly: NONE"""

# Few-shot beats instructions for a 1.5B model. None of these overlap the eval facts:
# an earlier prompt used an eval fact as its example and the model copied it everywhere.
EXTRACT_EXAMPLES = [
    ("The user said: How tall is Mount Everest?", "NONE"),
    (
        "The user said: My brother Imran just started a job at Infosys in Pune.",
        "- Has a brother called Imran\n- Imran works at Infosys in Pune",
    ),
    ("The user said: Thanks, that's really helpful!", "NONE"),
    (
        "The assistant had just said: What music do you like?\nThe user said: Mostly old ghazals.",
        "- Likes old ghazals",
    ),
    ("The user said: Can you recommend a good movie for tonight?", "NONE"),
    ("The user said: I've just moved to Kochi for good.", "- Lives in Kochi"),
]

# One pair per call: a 1.5B model judging a numbered list at once was unreliable (R2).
JUDGE_PROMPT = """You keep a memory of facts about a user. Compare an old fact with a newer one.
Reply with exactly one word:
- same: they say the same thing
- replaced: the newer fact means the old one is no longer true
- keep: both can be true at once, or they are about different things"""

JUDGE_EXAMPLES = [
    ("Old: Lives in Kolkata\nNew: Lives in Delhi", "replaced"),
    ("Old: Lives in Kolkata\nNew: Brother lives in Delhi", "keep"),
    ("Old: Has a son named Ayaan\nNew: Has a son called Ayaan", "same"),
    ("Old: Prefers beaches to mountains\nNew: Prefers mountains to beaches", "replaced"),
    ("Old: Likes tea\nNew: Likes cricket", "keep"),
    ("Old: Uses an Android phone\nNew: Bought an iPhone last month and uses it now", "replaced"),
    ("Old: Works as a teacher\nNew: Enjoys teaching chess on weekends", "keep"),
]


def build_system_prompt(memories: list[str]) -> str:
    if not memories:
        return SYSTEM_PROMPT
    remembered = "\n".join(f"- {m}" for m in memories)
    return f"{SYSTEM_PROMPT}\n\n{MEMORY_PREAMBLE}\n{remembered}"


def extraction_input(user_text: str, previous_reply: str) -> str:
    # The previous reply resolves answers like "Afifa" to "what is your sister called?".
    context = f"The assistant had just said: {previous_reply}\n" if previous_reply else ""
    return f"{context}The user said: {user_text}"


def judge_input(old_fact: str, new_fact: str) -> str:
    return f"Old: {old_fact}\nNew: {new_fact}"
