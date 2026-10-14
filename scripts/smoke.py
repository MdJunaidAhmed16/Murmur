"""Round-trips speech through TTS → VAD → STT. Checks the audio stages without a mic or vLLM."""

import asyncio
import sys

import numpy as np

from murmur.audio.frames import FRAME_SAMPLES, INPUT_SAMPLE_RATE, pcm16_to_float
from murmur.audio.vad import SileroVAD
from murmur.config import get_settings
from murmur.pipeline.metrics import now_ms
from murmur.stt.local import WhisperSTT
from murmur.tts.local import KokoroTTS

SENTENCE = "My sister is called Afifa and she plays badminton on weekday evenings."


async def main() -> int:
    settings = get_settings()
    tts = KokoroTTS(
        settings.models_dir, settings.tts_voice, settings.tts_speed, settings.tts_device
    )
    print(f"tts providers: {tts.providers}")
    stt = WhisperSTT(
        settings.stt_model, settings.stt_device, settings.stt_compute_type, settings.models_dir
    )
    vad = SileroVAD(settings.models_dir / "silero_vad.onnx")
    tts.warm_up()
    stt.warm_up()

    start = now_ms()
    first_chunk_ms = None
    chunks: list[bytes] = []
    async for pcm in tts.synthesize(SENTENCE):
        first_chunk_ms = first_chunk_ms or now_ms() - start
        chunks.append(pcm)
    audio_24k = pcm16_to_float(b"".join(chunks))
    print(
        f"tts: first chunk {first_chunk_ms:.0f} ms, {len(audio_24k) / tts.sample_rate:.1f}s audio"
    )

    n_out = int(len(audio_24k) * INPUT_SAMPLE_RATE / tts.sample_rate)
    audio = np.interp(
        np.linspace(0, len(audio_24k) - 1, n_out), np.arange(len(audio_24k)), audio_24k
    ).astype(np.float32)

    usable = len(audio) // FRAME_SAMPLES * FRAME_SAMPLES
    probs = [vad.speech_probability(f) for f in audio[:usable].reshape(-1, FRAME_SAMPLES)]
    speech = sum(p >= settings.vad_threshold for p in probs)
    print(f"vad: {speech}/{len(probs)} frames flagged as speech")

    start = now_ms()
    text = stt.transcribe(audio)
    print(f"stt: {now_ms() - start:.0f} ms on {settings.stt_device} → {text!r}")

    ok = speech > len(probs) // 2 and "badminton" in text.lower()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
