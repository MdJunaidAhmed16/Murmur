"""Silero VAD over ONNX, and the endpoint detector that turns speech flags into turns."""

from enum import Enum, auto
from pathlib import Path

import numpy as np
import onnxruntime as ort

from murmur.audio.frames import FRAME_MS, FRAME_SAMPLES, INPUT_SAMPLE_RATE, Float32Audio

# Silero v5 expects the previous 64 samples prepended to each 512-sample frame at 16 kHz.
_CONTEXT_SAMPLES = 64


class SileroVAD:
    """Stateful: feed frames in order, call reset() between sessions."""

    def __init__(self, model_path: Path) -> None:
        opts = ort.SessionOptions()
        # One thread is plenty for a 512-sample frame and keeps cores free for STT/TTS.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(INPUT_SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probability(self, frame: Float32Audio) -> float:
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {frame.shape[0]}")
        x = np.concatenate([self._context, frame.reshape(1, -1)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -_CONTEXT_SAMPLES:]
        return float(out[0][0])


class VadEvent(Enum):
    ONSET = auto()
    ABORT = auto()
    SPEECH_START = auto()
    ENDPOINT = auto()


class EndpointDetector:
    """Counts frames, not wall-clock, so replayed audio behaves exactly like live audio.

    Two thresholds for the start of speech: ONSET after onset_ms, cheap to act on and easy to
    take back (pause playback), and SPEECH_START after min_speech_ms, which commits to a turn.
    ABORT means an onset never became speech: a cough, a click, or the agent's own echo.
    """

    def __init__(self, endpoint_silence_ms: int, min_speech_ms: int, onset_ms: int) -> None:
        self._silence_frames_needed = max(1, endpoint_silence_ms // FRAME_MS)
        self._speech_frames_needed = max(1, min_speech_ms // FRAME_MS)
        self._onset_frames = max(1, onset_ms // FRAME_MS)
        self.in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    @property
    def _onset_fired(self) -> bool:
        return self._onset_frames <= self._speech_run < self._speech_frames_needed

    def update(self, is_speech: bool) -> VadEvent | None:
        if is_speech:
            self._silence_run = 0
            self._speech_run += 1
            if self.in_speech:
                return None
            if self._speech_run >= self._speech_frames_needed:
                self.in_speech = True
                return VadEvent.SPEECH_START
            if self._speech_run == self._onset_frames:
                return VadEvent.ONSET
            return None

        if not self.in_speech:
            aborted = self._onset_fired
            self._speech_run = 0
            return VadEvent.ABORT if aborted else None

        self._speech_run = 0
        if self.in_speech:
            self._silence_run += 1
            if self._silence_run >= self._silence_frames_needed:
                self.in_speech = False
                self._silence_run = 0
                return VadEvent.ENDPOINT
        return None
