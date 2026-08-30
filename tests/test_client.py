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


def test_compress_audio_outputs_small_mono_m4a(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "memo.m4a"
    source.write_bytes(b"input")
    seen_command: list[str] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        seen_command.extend(command)
        Path(command[-1]).write_bytes(b"m4a")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    compressed = client_module._compress_audio(source, tmp_path)

    assert compressed.suffix == ".m4a"
    assert seen_command[seen_command.index("-ac") + 1] == "1"
    assert seen_command[seen_command.index("-ar") + 1] == "16000"
    assert seen_command[seen_command.index("-c:a") + 1] == "aac"
    assert seen_command[seen_command.index("-b:a") + 1] == "32k"


def test_process_file_keeps_all_results_local(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"not-real-audio")

    monkeypatch.setattr(
        client_module,
        "_compress_audio",
        lambda source, destination: source,
    )
    monkeypatch.setattr(client_module, "wait_until_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        client_module,
        "_process_stream",
        lambda *args, save_transcript, **kwargs: (
            save_transcript(
                {
                    "text": "嗯，今天散步了。",
                    "chunks": [{"text": "嗯，今天散步了。", "chunk_index": 0}],
                }
            )
            or "嗯，今天散步了。"
        ),
    )
    monkeypatch.setattr(
        client_module,
        "_rewrite_long_transcript",
        lambda *args, **kwargs: {
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
    )

    assert (output / "diary.md").read_text("utf-8") == "# 今天\n\n今天散步了。\n"
    raw = json.loads((output / "raw_transcript.json").read_text("utf-8"))
    assert raw["text"] == "嗯，今天散步了。"
    assert not (output / "cleaned_transcript.md").exists()
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
        "_compress_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ASR reran")),
    )
    monkeypatch.setattr(
        client_module,
        "_rewrite_long_transcript",
        lambda *args, **kwargs: {
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
    )

    assert output == checkpoint
    assert (output / "diary.md").exists()


def test_process_file_uses_checkpoint_if_stream_drops_after_asr(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"not-real-audio")
    output_root = tmp_path / "output"
    calls = {"process": 0, "rewrite": 0}

    monkeypatch.setattr(client_module, "_compress_audio", lambda source, destination: source)
    monkeypatch.setattr(client_module, "wait_until_ready", lambda *args, **kwargs: None)

    def dropped_stream(*args, save_transcript, **kwargs):  # type: ignore[no-untyped-def]
        calls["process"] += 1
        save_transcript({"text": "已经完成的 ASR", "chunks": []})
        raise client_module.StreamEnded("connection dropped")

    def rewrite(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["rewrite"] += 1
        return {"diary": "# 日记\n\n已经完成的 ASR", "uncertainties": []}

    monkeypatch.setattr(client_module, "_process_stream", dropped_stream)
    monkeypatch.setattr(client_module, "_rewrite_long_transcript", rewrite)

    output = client_module.process_file(
        audio,
        endpoint_url="https://example.invalid",
        api_key="secret",
        app_token=None,
        output_root=output_root,
        language="Chinese",
        date=None,
        title_hint=None,
    )

    assert calls == {"process": 1, "rewrite": 1}
    assert (output / "diary.md").exists()


def test_long_transcript_is_rewritten_in_bounded_parts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def rewrite(client, base_url, headers, transcript, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(transcript)
        return {"diary": f"# 日记\n\n{transcript}", "uncertainties": []}

    monkeypatch.setattr(client_module, "_rewrite_with_one_retry", rewrite)
    result = client_module._rewrite_long_transcript(
        object(),
        "https://example.invalid",
        {},
        "第一段。" * 1000,
        date=None,
        title_hint=None,
    )

    assert len(calls) > 1
    assert all(len(part) <= 2400 for part in calls)
    assert result["diary"].count("# 日记") == 1
