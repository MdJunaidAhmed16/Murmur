# Replay clips

Stand-in voice data until real recordings exist. WAVs are gitignored; `manifest.json`
lists each clip with its dataset transcript.

Source: 10 en-US utterances from **MInDS-14** (PolyAI), CC-BY-4.0 —
Gerz et al., "Multilingual and Cross-Lingual Intent Detection from Spoken Data", 2021.
https://huggingface.co/datasets/PolyAI/minds14

Real callers talking to a banking assistant over the phone: 8 kHz μ-law, fillers,
mid-sentence pauses. Harder than a laptop mic, which makes it a fair stress test.
The dataset's own transcripts are ASR output and contain errors.

```bash
uv run python scripts/replay.py clips/*.wav
```
