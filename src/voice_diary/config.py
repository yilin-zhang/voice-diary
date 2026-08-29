from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_temp_dir() -> Path:
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
        return shared_memory / "voice-diary"
    return Path(tempfile.gettempdir()) / "voice-diary"


@dataclass(frozen=True, slots=True)
class Settings:
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str | None = "Qwen/Qwen3-ForcedAligner-0.6B"
    editor_model: str = "Qwen/Qwen3-14B-FP8"
    max_upload_bytes: int = 28 * 1024 * 1024
    max_editor_tokens: int = 8192
    temp_dir: Path = _default_temp_dir()
    app_token: str | None = None
    fake_mode: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        aligner = os.getenv(
            "VOICE_DIARY_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"
        ).strip()
        return cls(
            asr_model=os.getenv("VOICE_DIARY_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
            aligner_model=aligner or None,
            editor_model=os.getenv(
                "VOICE_DIARY_EDITOR_MODEL", "Qwen/Qwen3-14B-FP8"
            ),
            max_upload_bytes=int(
                os.getenv("VOICE_DIARY_MAX_UPLOAD_BYTES", str(28 * 1024 * 1024))
            ),
            max_editor_tokens=int(os.getenv("VOICE_DIARY_MAX_EDITOR_TOKENS", "8192")),
            temp_dir=Path(os.getenv("VOICE_DIARY_TEMP_DIR", str(_default_temp_dir()))),
            app_token=os.getenv("VOICE_DIARY_APP_TOKEN") or None,
            fake_mode=_env_bool("VOICE_DIARY_FAKE_MODE"),
        )
