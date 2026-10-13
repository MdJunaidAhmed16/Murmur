"""Pulls VAD, STT, TTS and LLM weights. Safe to re-run; skips what is already present."""

import urllib.request
from pathlib import Path

from faster_whisper import download_model
from huggingface_hub import snapshot_download

from murmur.config import get_settings

FILES = {
    "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"have  {dest}")
        return
    print(f"fetch {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        fetch(url, settings.models_dir / name)

    print(f"fetch whisper {settings.stt_model}")
    download_model(settings.stt_model, cache_dir=str(settings.models_dir))

    # vLLM reads from the Hugging Face cache, so the LLM goes there rather than models/.
    print(f"fetch {settings.llm_model}")
    snapshot_download(settings.llm_model)
    print("done")


if __name__ == "__main__":
    main()
