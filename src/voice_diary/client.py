from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


class ClientError(RuntimeError):
    pass


def _runpod_api_key(config_path: Path | None = None) -> str | None:
    environment_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if environment_key:
        return environment_key

    path = config_path or Path.home() / ".runpod" / "config.toml"
    try:
        config = tomllib.loads(path.read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    profile_name = os.getenv("RUNPOD_PROFILE", "default")
    profile = config.get(profile_name, {})
    candidates = [
        profile.get("api_key") if isinstance(profile, dict) else None,
        config.get("api_key"),
        config.get("apikey"),
    ]
    return next(
        (value.strip() for value in candidates if isinstance(value, str) and value.strip()),
        None,
    )


def _headers(api_key: str, app_token: str | None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if app_token:
        headers["X-Voice-Diary-Key"] = app_token
    return headers


def wait_until_ready(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    *,
    attempts: int = 30,
) -> None:
    for attempt in range(attempts):
        try:
            response = client.get(f"{base_url}/ping", headers=headers, timeout=15)
            if response.status_code == 200:
                return
            if response.status_code not in {204, 502, 503}:
                response.raise_for_status()
        except httpx.TransportError:
            pass
        time.sleep(min(5 + attempt, 15))
    raise ClientError("endpoint did not become ready")


def _split_audio(source: Path, destination: Path, seconds: int) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise ClientError("ffmpeg is required to split large audio files")
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
        raise ClientError("ffmpeg could not split the audio")
    chunks = sorted(destination.glob("chunk-*.wav"))
    if not chunks:
        raise ClientError("ffmpeg produced no audio chunks")
    return chunks


def _transcribe(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    audio: Path,
    language: str,
) -> dict[str, Any]:
    content_type = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    with audio.open("rb") as stream:
        response = client.post(
            f"{base_url}/v1/transcribe",
            headers=headers,
            files={"audio": ("audio" + audio.suffix, stream, content_type)},
            data={"language": language},
            timeout=330,
        )
    if response.status_code == 413:
        raise ClientError("audio chunk is still too large; reduce --chunk-seconds")
    response.raise_for_status()
    return response.json()


def _rewrite(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    transcript: str,
    *,
    date: str | None,
    title_hint: str | None,
) -> dict[str, Any]:
    with client.stream(
        "POST",
        f"{base_url}/v1/rewrite-stream",
        headers=headers,
        json={"transcript": transcript, "date": date, "title_hint": title_hint},
        timeout=330,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                return json.loads(line.removeprefix("data: "))
    raise ClientError("rewrite stream ended without a result")


def process_file(
    audio: Path,
    *,
    endpoint_url: str,
    api_key: str,
    app_token: str | None,
    output_root: Path,
    language: str,
    date: str | None,
    title_hint: str | None,
    chunk_seconds: int,
) -> Path:
    if not audio.is_file():
        raise ClientError(f"audio file does not exist: {audio}")

    headers = _headers(api_key, app_token)
    base_url = endpoint_url.rstrip("/")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / f"{audio.stem}-{stamp}"
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix="voice-diary-chunks-") as raw_temp:
        temp_dir = Path(raw_temp)
        chunks = _split_audio(audio, temp_dir, chunk_seconds)
        transcripts: list[dict[str, Any]] = []

        with httpx.Client() as client:
            wait_until_ready(client, base_url, headers)
            for index, chunk in enumerate(chunks):
                result = _transcribe(client, base_url, headers, chunk, language)
                result["chunk_index"] = index
                result["offset_seconds"] = index * chunk_seconds
                transcripts.append(result)

            full_transcript = "\n".join(item["text"].strip() for item in transcripts).strip()
            (output_dir / "raw_transcript.json").write_text(
                json.dumps(
                    {"source": audio.name, "chunks": transcripts, "text": full_transcript},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            rewrite = _rewrite(
                client,
                base_url,
                headers,
                full_transcript,
                date=date,
                title_hint=title_hint,
            )

    (output_dir / "cleaned_transcript.md").write_text(
        rewrite["cleaned_transcript"].rstrip() + "\n", encoding="utf-8"
    )
    (output_dir / "diary.md").write_text(rewrite["diary"].rstrip() + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "processed_at": datetime.now().astimezone().isoformat(),
                "language": language,
                "date": date,
                "title_hint": title_hint,
                "uncertainties": rewrite.get("uncertainties", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe a memo into a private local diary")
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("VOICE_DIARY_ENDPOINT_URL"),
        help="Runpod load-balanced endpoint base URL",
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--date")
    parser.add_argument("--title")
    parser.add_argument("--chunk-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path(".voice-diary-output"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    api_key = _runpod_api_key()
    if not api_key:
        sys.exit("Runpod API key not found; run `runpodctl doctor` or set RUNPOD_API_KEY")
    if not args.endpoint:
        sys.exit("--endpoint or VOICE_DIARY_ENDPOINT_URL is required")
    try:
        output_dir = process_file(
            args.audio,
            endpoint_url=args.endpoint,
            api_key=api_key,
            app_token=os.getenv("VOICE_DIARY_APP_TOKEN"),
            output_root=args.output,
            language=args.language,
            date=args.date,
            title_hint=args.title,
            chunk_seconds=args.chunk_seconds,
        )
    except (ClientError, httpx.HTTPError) as exc:
        sys.exit(f"voice-diary failed: {exc}")
    print(output_dir)


if __name__ == "__main__":
    main()
