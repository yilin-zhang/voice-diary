from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import UploadFile


class UploadTooLarge(ValueError):
    pass


_CONTENT_TYPE_SUFFIXES = {
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "application/octet-stream": ".audio",
}


def _suffix_for(upload: UploadFile) -> str:
    return _CONTENT_TYPE_SUFFIXES.get((upload.content_type or "").lower(), ".audio")


@asynccontextmanager
async def private_upload(
    upload: UploadFile,
    *,
    directory: Path,
    max_bytes: int,
    chunk_bytes: int = 1024 * 1024,
) -> AsyncIterator[tuple[Path, int]]:
    """Persist an upload under a random name and guarantee unlinking on every exit path."""

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="request-", suffix=_suffix_for(upload), dir=directory)
    path = Path(raw_path)
    total = 0

    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "wb") as destination:
            while chunk := await upload.read(chunk_bytes):
                total += len(chunk)
                if total > max_bytes:
                    raise UploadTooLarge(f"upload exceeds {max_bytes} bytes")
                destination.write(chunk)
        yield path, total
    finally:
        await upload.close()
        path.unlink(missing_ok=True)
