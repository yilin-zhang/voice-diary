from __future__ import annotations

import json
from pathlib import Path

from voice_diary import client as client_module


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
