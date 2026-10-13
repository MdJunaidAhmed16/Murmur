"""The one Settings object, read from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MURMUR_", env_file=".env", extra="ignore")

    backend: Literal["local", "hosted"] = "local"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_max_tokens: int = 256

    stt_model: str = "small.en"
    stt_device: Literal["cuda", "cpu"] = "cuda"
    stt_compute_type: str = "float16"
    # Re-transcribe the growing utterance this often to emit partials.
    stt_partial_interval_ms: int = 600

    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    # "complete" waits for the whole reply; kept so the handoff gain can be measured (docs/01).
    tts_handoff: Literal["sentence", "complete"] = "sentence"

    vad_threshold: float = 0.5
    # The endpointing trade-off from docs/01: lower cuts people off, higher feels sluggish.
    endpoint_silence_ms: int = 200
    min_speech_ms: int = 250
    # Off if the agent hears itself through laptop speakers and interrupts its own replies.
    barge_in: bool = True

    working_memory_turns: int = 12

    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
