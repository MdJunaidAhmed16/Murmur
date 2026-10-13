"""The wire format between browser and server, and PCM conversions."""

import numpy as np
import numpy.typing as npt

# Browser sends 16 kHz mono int16; 512 samples is the frame size Silero VAD expects at 16 kHz.
INPUT_SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // INPUT_SAMPLE_RATE

Float32Audio = npt.NDArray[np.float32]
Int16Audio = npt.NDArray[np.int16]


def pcm16_to_float(data: bytes) -> Float32Audio:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(audio: Float32Audio) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
