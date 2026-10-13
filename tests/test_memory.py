from datetime import timedelta
from pathlib import Path

from murmur.memory.records import utc_now
from tests.fakes import make_memory


def test_recall_ranks_relevant_fact_first_and_skips_unrelated(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    memory.add(["Plays badminton on weekday evenings", "Has a sister called Afifa"], None)

    hits = memory.recall("which evenings for badminton", k=4)

    assert [m.text for m in hits] == ["Plays badminton on weekday evenings"]


def test_memories_survive_reopening_the_store(tmp_path: Path) -> None:
    make_memory(tmp_path / "m.db").add(["Lives in Chennai"], "turn_1")

    reopened = make_memory(tmp_path / "m.db")

    [hit] = reopened.recall("lives in which city", k=1)
    assert hit.text == "Lives in Chennai"
    assert hit.source_turn == "turn_1"


def test_superseded_facts_are_kept_but_not_recalled(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    [old] = memory.add(["Lives in Chennai"], None)
    [new] = memory.add(["Lives in Bangalore"], None)

    memory.supersede(old.id, new.id)

    assert [m.text for m in memory.recall("which city, lives in", k=4)] == ["Lives in Bangalore"]
    kept = memory.get(old.id)
    assert kept is not None and kept.superseded_by == new.id


def test_forget_hard_deletes_the_match_and_what_it_superseded(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    [old] = memory.add(["Works at Infosys"], None)
    [new] = memory.add(["Works at Google"], None)
    [other] = memory.add(["Has a cat called Miso"], None)
    memory.supersede(old.id, new.id)

    removed = memory.forget("works at Google", threshold=0.6)

    assert {m.text for m in removed} == {"Works at Google", "Works at Infosys"}
    assert [m.id for m in memory.all(include_superseded=True)] == [other.id]
    assert make_memory(tmp_path / "m.db").get(old.id) is None  # gone from disk, not just cache


def test_decay_demotes_stale_facts_below_fresh_ones(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    [stale] = memory.add(["Likes green tea in the morning"], None)
    [fresh] = memory.add(["Likes black tea in the evening"], None)
    memory.mark_used([fresh.id])

    decayed = memory.decay(timedelta(days=30), now=utc_now() + timedelta(days=31))

    assert decayed == 2
    memory.mark_used([fresh.id])  # use revives priority
    hits = memory.recall("tea", k=2)  # equally similar to both: only priority separates them
    assert hits[0].id == fresh.id
    assert {m.id for m in hits} == {stale.id, fresh.id}


def test_mark_used_counts_and_recall_has_no_side_effects(tmp_path: Path) -> None:
    memory = make_memory(tmp_path / "m.db")
    [fact] = memory.add(["Drives a red Swift"], None)

    memory.recall("what car do I drive", k=1)
    unchanged = memory.get(fact.id)
    memory.mark_used([fact.id])
    used = memory.get(fact.id)

    assert unchanged is not None and unchanged.use_count == 0
    assert used is not None and used.use_count == 1 and used.last_used_at is not None
