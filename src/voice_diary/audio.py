from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AudioConversionError(RuntimeError):
    pass


def split_audio(source: Path, destination: Path, seconds: int) -> list[Path]:
    """Decode an uploaded memo into portable mono 16 kHz PCM chunks."""
    if shutil.which("ffmpeg") is None:
        raise AudioConversionError("ffmpeg is required")
    pattern = destination / "chunk-%05d.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_format",
        "wav",
        "-segment_time",
        str(seconds),
        "-reset_timestamps",
        "1",
        str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioConversionError("ffmpeg could not decode the audio")
    chunks = sorted(destination.glob("chunk-*.wav"))
    if not chunks:
        raise AudioConversionError("ffmpeg produced no audio chunks")
    return chunks
