from __future__ import annotations

import json
import subprocess
from pathlib import Path

from voice_diary import client as client_module


def test_runpod_api_key_prefers_environment(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "config.toml"
    config.write_text('[default]\napi_key = "from-file"\n', encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "from-environment")

    assert client_module._runpod_api_key(config) == "from-environment"


def test_runpod_api_key_reads_default_profile(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "config.toml"
    config.write_text('[default]\napi_key = "from-file"\n', encoding="utf-8")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    assert client_module._runpod_api_key(config) == "from-file"


def test_runpod_api_key_handles_missing_or_invalid_config(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert client_module._runpod_api_key(tmp_path / "missing.toml") is None
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("not = [valid", encoding="utf-8")
    assert client_module._runpod_api_key(invalid) is None


def test_split_audio_outputs_asr_compatible_wav(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "memo.m4a"
    source.write_bytes(b"input")
    seen_command: list[str] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        seen_command.extend(command)
        Path(command[-1].replace("%05d", "00000")).write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    chunks = client_module._split_audio(source, tmp_path, 600)

    assert [chunk.suffix for chunk in chunks] == [".wav"]
    assert seen_command[seen_command.index("-c:a") + 1] == "pcm_s16le"
    assert seen_command[seen_command.index("-segment_format") + 1] == "wav"


def test_process_file_keeps_all_results_local(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"not-real-audio")

    monkeypatch.setattr(
        client_module,
        "_split_audio",
        lambda source, destination, seconds: [source],
    )
    monkeypatch.setattr(client_module, "wait_until_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        client_module,
        "_transcribe",
        lambda *args, **kwargs: {
            "text": "嗯，今天散步了。",
            "language": "Chinese",
            "timestamps": [],
        },
    )
    monkeypatch.setattr(
        client_module,
        "_rewrite",
        lambda *args, **kwargs: {
            "cleaned_transcript": "今天散步了。",
            "diary": "# 今天\n\n今天散步了。",
            "uncertainties": [],
        },
    )

    output = client_module.process_file(
        audio,
        endpoint_url="https://example.invalid",
        api_key="not-written-to-disk",
        app_token=None,
        output_root=tmp_path / "output",
        language="Chinese",
        date="2026-08-28",
        title_hint=None,
        chunk_seconds=600,
    )

    assert (output / "diary.md").read_text("utf-8") == "# 今天\n\n今天散步了。\n"
    raw = json.loads((output / "raw_transcript.json").read_text("utf-8"))
    assert raw["text"] == "嗯，今天散步了。"
    assert "not-written-to-disk" not in "".join(
        path.read_text("utf-8") for path in output.iterdir()
    )


def test_process_file_reuses_unfinished_transcript(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"not-real-audio")
    output_root = tmp_path / "output"
    checkpoint = output_root / "memo-20260828-120000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "raw_transcript.json").write_text(
        json.dumps({"source": "memo.m4a", "chunks": [], "text": "已完成的转录"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(client_module, "wait_until_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        client_module,
        "_split_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ASR reran")),
    )
    monkeypatch.setattr(
        client_module,
        "_rewrite",
        lambda *args, **kwargs: {
            "cleaned_transcript": "已完成的转录",
            "diary": "# 日记\n\n已完成的转录",
            "uncertainties": [],
        },
    )

    output = client_module.process_file(
        audio,
        endpoint_url="https://example.invalid",
        api_key="secret",
        app_token=None,
        output_root=output_root,
        language="Chinese",
        date=None,
        title_hint=None,
        chunk_seconds=60,
    )

    assert output == checkpoint
    assert (output / "diary.md").exists()
