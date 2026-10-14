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
    # "clause": the first chunk may end at a comma, so a long opening sentence starts sooner.
    tts_first_chunk: Literal["sentence", "clause"] = "clause"
    # "cuda" runs Kokoro on the GPU (onnxruntime-gpu, CUDA 12). CPU until R3 measures both.
    tts_device: Literal["cpu", "cuda"] = "cpu"

    vad_threshold: float = 0.5
    # Speculative endpointing (R3). The turn starts after endpoint_silence_ms of silence, but its
    # first audio is held until endpoint_commit_ms: if the user carries on in between, the turn
    # is cancelled unheard and their words merge into the next one. On the clip set, 200 ms cut
    # 7 of 16 utterances mid-way and 600 ms cut none (longest inner pause 544 ms).
    endpoint_silence_ms: int = 200
    endpoint_commit_ms: int = 600
    min_speech_ms: int = 250
    # Off if the agent hears itself through laptop speakers and interrupts its own replies.
    barge_in: bool = True
    # Speech this long while the agent is audible pauses playback; min_speech_ms confirms it as
    # an interruption (clear) or not (resume). Gate 3 wants the cut inside 200 ms.
    barge_in_onset_ms: int = 96

    working_memory_turns: int = 12

    memory_db: Path = Path("data/memory.db")
    embed_model: str = "BAAI/bge-small-en-v1.5"
    recall_k: int = 4
    # 40ms ceiling: recall sits inside the 250ms model budget (see docs/01).
    recall_budget_ms: int = 40
    # bge similarities are compressed: question-to-fact matches scored 0.53-0.76 on the eval,
    # so the floor only drops the clearly unrelated and top-k ranking does the rest (R2).
    recall_min_similarity: float = 0.5
    forget_threshold: float = 0.7
    # Embeddings barely see word order or negation ("tea over coffee" vs "coffee over tea"),
    # so only near-identical text merges without asking the judge.
    memory_duplicate_threshold: float = 0.98
    # Only pairs this similar reach the LLM judge. On the eval, true contradictions scored
    # 0.64-0.997 and the judge's false supersedes 0.61-0.71: the 1.5B judge says "replaced"
    # too readily, so the gate does most of the precision work (R2).
    memory_related_threshold: float = 0.7
    memory_stale_days: int = 30

    # For the current_time tool when the user names no place.
    timezone: str = "Asia/Kolkata"

    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
