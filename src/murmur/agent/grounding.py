"""Content words: what a sentence is about once the words every fact shares are removed."""

import re

# Words every fact uses; sharing one of these with the user proves nothing.
_FRAME_WORDS = frozenset(
    "the user users has have had is are was were and for with their they them who what that"
    " this likes like prefers prefer enjoys enjoy loves love lives live works work called named"
    " currently usually also very really from about into onto your you"
    " include includes".split()
)


def content_words(text: str) -> set[str]:
    words = (w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()))
    return {w for w in words if len(w) >= 3 and w not in _FRAME_WORDS}


def shares_content(a: str, b: str) -> bool:
    return bool(content_words(a) & content_words(b))
