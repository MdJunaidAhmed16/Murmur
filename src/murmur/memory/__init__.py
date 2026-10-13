"""Long-term memory: atomic facts that survive the session.

The critical-path rule (docs/02):

    TURN:  recall (40ms, blocking)  ->  generate  ->  speak
    AFTER: extract and write (async)
    LATER: consolidate (batched)

Nothing in this package may run on the turn path except MemoryService.recall, which is a pure
read of an in-memory snapshot. Adding "just one quick write" to the turn is always tempting and
always wrong. This package never calls the model; it is given text and returns text.
"""

from murmur.memory.records import Memory
from murmur.memory.service import MemoryService, open_memory

__all__ = ["Memory", "MemoryService", "open_memory"]
