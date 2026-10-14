"""Writes the README's latency and memory numbers from measured results. Never type them.

    uv run python scripts/readme_numbers.py \
        --latency results/r3/tuned/latency.json results/r3/hosted/latency.json

Replaces what sits between the <!-- latency:start/end --> and <!-- memory:start/end -->
markers. A backend or file with no data renders as TBD, never as a guess.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

README = Path(__file__).resolve().parent.parent / "README.md"

# (stage key, label, budget from docs/01)
STAGES = [
    ("endpoint", "Endpointing", "200 ms"),
    ("stt_final", "Transcription (final)", "100 ms"),
    ("memory_recall", "Memory recall", "40 ms"),
    ("llm_first_token", "Model, first token", "250 ms"),
    ("llm_first_sentence", "Model, first chunk", "—"),
    ("tts_first_audio", "Speech, first audio", "200 ms"),
    ("time_to_first_audio", "**Time to first audio**", "**800 ms**"),
]
BACKENDS = [("local", "Local (RTX 4060 Laptop)"), ("hosted", "Hosted")]


def cell(stats: dict[str, Any] | None, stage: str) -> str:
    s = (stats or {}).get(stage) or {}
    if s.get("p50") is None:
        return "TBD"
    text = f"{s['p50']:.0f} / {s['p90']:.0f} ms"
    return f"**{text}**" if stage == "time_to_first_audio" else text


def latency_block(report: dict[str, Any] | None, source: str, barge: dict[str, Any] | None) -> str:
    backends = (report or {}).get("backends", {})
    head = "| Stage | " + " | ".join(label for _, label in BACKENDS) + " | Budget |"
    rows = [head, "| --- |" + " --- |" * (len(BACKENDS) + 1)]
    for stage, label, budget in STAGES:
        cells = " | ".join(cell(backends.get(key), stage) for key, _ in BACKENDS)
        rows.append(f"| {label} | {cells} | {budget} |")
    if report is None:
        return "\n".join(rows) + "\n\np50 / p90. Not yet measured."
    stack = report["stack"]
    turns = {k: v["time_to_first_audio"]["n"] for k, v in backends.items()}
    hosted = f"Hosted: {stack['hosted_model']}" if "hosted" in backends else "Hosted: TBD"
    cut = cell((barge or {}).get("backends", {}).get("local"), "barge_in")
    note = (
        f"p50 / p90, from {source} ({report['generated_at']}); answered turns per backend: "
        f"{', '.join(f'{k} {n}' for k, n in turns.items())}. Local: {stack['llm_model']} on "
        f"vLLM, {stack['gpu']}. {hosted}. Interrupting it stops audible playback in {cut} "
        "(first speech frame to stop command; `reports/R3-latency.md`)."
    )
    return "\n".join(rows) + "\n\n" + note


def memory_block(report: dict[str, Any] | None, source: str) -> str:
    if report is None:
        return "**TBD** — recall precision, store size per 100 turns, contradiction handling."
    per100 = report["records_per_100_turns"]
    contra = report["contradictions"]
    return "\n".join(
        [
            "| Check | Development set | Held-out set |",
            "| --- | --- | --- |",
            f"| Recall precision (fact in top {report['settings']['recall_k']}) | "
            f"{report['dev']['recall_hit_rate']} | {report['heldout']['recall_hit_rate']} |",
            f"| Spoken answer uses the fact | {report['dev']['answer_uses_fact']} | "
            f"{report['heldout']['answer_uses_fact']} |",
            f"| Contradiction: old fact superseded | {contra['dev']['old_superseded']} | "
            f"{contra['heldout']['old_superseded']} |",
            "",
            f"Store growth: {per100['before']:.0f} records per 100 turns before consolidation, "
            f"{per100['after']:.1f} after. Recall {report['recall_ms']['p50']:.0f} ms p50 against "
            f"a 40 ms budget. From `{source}`; scripted facts, not a benchmark — "
            "`reports/R2-memory.md` has what was tuned and what the held-out set shows.",
        ]
    )


def replace_between(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:start -->).*?(<!-- {name}:end -->)", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"README is missing the <!-- {name}:start/end --> markers")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Several files merge by backend: e.g. results/r3/tuned/... (local) + results/r3/hosted/...
    parser.add_argument("--latency", type=Path, nargs="+", default=[Path("results/latency.json")])
    parser.add_argument("--memory", type=Path, default=Path("results/memory_eval/report.json"))
    parser.add_argument("--barge-in", type=Path, default=Path("results/r3/barge-in/latency.json"))
    args = parser.parse_args()

    text = README.read_text()
    reports = [r for r in (load(p) for p in args.latency) if r is not None]
    merged = None
    if reports:
        merged = {**reports[0], "backends": {}}
        for r in reports:
            merged["backends"].update(r["backends"])
            # Whichever run used the hosted backend knows its model.
            if "hosted" in r["backends"]:
                merged["stack"] = {**merged["stack"], "hosted_model": r["stack"]["hosted_model"]}
    sources = " + ".join(f"`{p}`" for p in args.latency if p.exists())
    latency = latency_block(merged, sources, load(args.barge_in))
    text = replace_between(text, "latency", latency)
    text = replace_between(text, "memory", memory_block(load(args.memory), str(args.memory)))
    README.write_text(text)
    print(f"updated {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
