from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


class ClientError(RuntimeError):
    pass


class StreamEnded(ClientError):
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
    headers = {"Authorization": f"Bearer {api_key}", "Connection": "close"}
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


def _compress_audio(source: Path, destination: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise ClientError("ffmpeg is required to compress audio")
    target = destination / "memo.m4a"
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
        "aac",
        "-b:a",
        "32k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ClientError("ffmpeg could not compress the audio")
    if not target.is_file() or target.stat().st_size == 0:
        raise ClientError("ffmpeg produced no compressed audio")
    return target


def _events(response: httpx.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    event = "message"
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line.startswith(":"):
            continue
        if not line:
            if data_lines:
                yield event, json.loads("\n".join(data_lines))
            event = "message"
            data_lines = []
        elif line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
    if data_lines:
        yield event, json.loads("\n".join(data_lines))


def _process_stream(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    audio: Path,
    *,
    language: str,
    date: str | None,
    title_hint: str | None,
    save_transcript: Callable[[dict[str, Any]], None],
) -> str:
    timeout = httpx.Timeout(connect=30, read=None, write=120, pool=30)
    with (
        audio.open("rb") as stream,
        client.stream(
            "POST",
            f"{base_url}/v1/process-stream",
            headers=headers,
            files={"audio": ("audio.m4a", stream, "audio/mp4")},
            data={
                "language": language,
                "date": date or "",
                "title_hint": title_hint or "",
            },
            timeout=timeout,
        ) as response,
    ):
        if response.status_code == 413:
            raise ClientError("compressed audio exceeds the server upload limit")
        response.raise_for_status()
        for event, payload in _events(response):
            if event == "error" or payload.get("error"):
                raise ClientError("voice processing failed")
            if event == "transcript":
                save_transcript(payload)
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    raise StreamEnded("processing stream ended without a transcript")


def _rewrite(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    transcript: str,
    *,
    date: str | None,
    title_hint: str | None,
) -> dict[str, Any]:
    # Diary generation can legitimately take several minutes. Keep connection,
    # write and pool timeouts bounded, but let the SSE stream run until the
    # server returns its final result.
    timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
    with client.stream(
        "POST",
        f"{base_url}/v1/rewrite-stream",
        headers=headers,
        json={"transcript": transcript, "date": date, "title_hint": title_hint},
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for event, payload in _events(response):
            if event == "error" or payload.get("error"):
                raise ClientError("diary refinement failed")
            if event == "diary":
                return payload
    raise StreamEnded("rewrite stream ended without a diary")


def _unfinished_output(audio: Path, output_root: Path) -> tuple[Path, str] | None:
    """Reuse the newest local ASR checkpoint after a failed rewrite."""
    for candidate in sorted(output_root.glob(f"{audio.stem}-*"), reverse=True):
        checkpoint = candidate / "raw_transcript.json"
        if (candidate / "diary.md").exists() or not checkpoint.is_file():
            continue
        try:
            payload = json.loads(checkpoint.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = payload.get("text")
        if payload.get("source") == audio.name and isinstance(text, str) and text.strip():
            return candidate, text.strip()
    return None


def _retryable(exc: Exception, *, allow_bad_request: bool = False) -> bool:
    if isinstance(exc, (httpx.TransportError, StreamEnded)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        statuses = {502, 503, 504}
        if allow_bad_request:
            statuses.add(400)
        return exc.response.status_code in statuses
    return False


def _rewrite_with_one_retry(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    transcript: str,
    *,
    date: str | None,
    title_hint: str | None,
) -> dict[str, Any]:
    for attempt in range(2):
        try:
            return _rewrite(
                client,
                base_url,
                headers,
                transcript,
                date=date,
                title_hint=title_hint,
            )
        except Exception as exc:
            if attempt or not _retryable(exc):
                raise
            time.sleep(2)
            wait_until_ready(client, base_url, headers)
    raise AssertionError("unreachable")


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
) -> Path:
    if not audio.is_file():
        raise ClientError(f"audio file does not exist: {audio}")

    headers = _headers(api_key, app_token)
    base_url = endpoint_url.rstrip("/")
    checkpoint = _unfinished_output(audio, output_root)
    if checkpoint is not None:
        output_dir, full_transcript = checkpoint
        with httpx.Client(limits=httpx.Limits(max_keepalive_connections=0)) as client:
            wait_until_ready(client, base_url, headers)
            diary = _rewrite_with_one_retry(
                client,
                base_url,
                headers,
                full_transcript,
                date=date,
                title_hint=title_hint,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="voice-diary-upload-") as raw_temp:
            compressed = _compress_audio(audio, Path(raw_temp))
            if compressed.stat().st_size > 27 * 1024 * 1024:
                raise ClientError("compressed audio exceeds the safe upload limit")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = output_root / f"{audio.stem}-{stamp}"
            output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
            transcript_path = output_dir / "raw_transcript.json"

            def save_transcript(payload: dict[str, Any]) -> None:
                transcript_path.write_text(
                    json.dumps(
                        {"source": audio.name, **payload},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            with httpx.Client(limits=httpx.Limits(max_keepalive_connections=0)) as client:
                wait_until_ready(client, base_url, headers)
                for attempt in range(2):
                    try:
                        full_transcript = _process_stream(
                            client,
                            base_url,
                            headers,
                            compressed,
                            language=language,
                            date=date,
                            title_hint=title_hint,
                            save_transcript=save_transcript,
                        )
                        diary = _rewrite_with_one_retry(
                            client,
                            base_url,
                            headers,
                            full_transcript,
                            date=date,
                            title_hint=title_hint,
                        )
                        break
                    except Exception as exc:
                        saved = _unfinished_output(audio, output_root)
                        if saved is not None:
                            output_dir, full_transcript = saved
                            diary = _rewrite_with_one_retry(
                                client,
                                base_url,
                                headers,
                                full_transcript,
                                date=date,
                                title_hint=title_hint,
                            )
                            break
                        if attempt or not _retryable(exc, allow_bad_request=True):
                            raise
                        time.sleep(2)
                        wait_until_ready(client, base_url, headers)
                else:
                    raise AssertionError("unreachable")

    (output_dir / "diary.md").write_text(diary["diary"].rstrip() + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "processed_at": datetime.now().astimezone().isoformat(),
                "language": language,
                "date": date,
                "title_hint": title_hint,
                "uncertainties": diary.get("uncertainties", []),
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
        )
    except (ClientError, httpx.HTTPError) as exc:
        sys.exit(f"voice-diary failed: {exc}")
    print(output_dir)


if __name__ == "__main__":
    main()
