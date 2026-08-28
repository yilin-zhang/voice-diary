from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

from voice_diary.app import create_app
from voice_diary.config import Settings
from voice_diary.models import FakeBackend, ModelManager


def ready_manager(backend=None) -> ModelManager:
    manager = ModelManager(backend or FakeBackend())
    manager.load()
    return manager


def test_ping_and_rewrite(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(temp_dir=tmp_path),
        manager=ready_manager(),
        start_loader=False,
    )
    with TestClient(app) as client:
        assert client.get("/ping").json() == {"status": "healthy"}
        response = client.post(
            "/v1/rewrite",
            json={"transcript": "嗯，今天散步了。", "date": "2026-08-28"},
        )
    assert response.status_code == 200
    assert response.json()["cleaned_transcript"] == "今天散步了。"
    assert response.headers["cache-control"] == "no-store"


def test_transcribe_unlinks_private_upload(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(temp_dir=tmp_path),
        manager=ready_manager(),
        start_loader=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/transcribe",
            files={"audio": ("private-name.m4a", b"private audio bytes", "audio/mp4")},
            data={"language": "Chinese"},
        )
    assert response.status_code == 200
    assert response.json()["language"] == "Chinese"
    assert list(tmp_path.iterdir()) == []


def test_too_large_upload_is_rejected_and_unlinked(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(temp_dir=tmp_path, max_upload_bytes=4),
        manager=ready_manager(),
        start_loader=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/transcribe",
            files={"audio": ("secret.m4a", b"12345", "audio/mp4")},
        )
    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


class LeakyFailureBackend(FakeBackend):
    def transcribe(self, audio_path: Path, language: str | None):  # type: ignore[no-untyped-def]
        raise RuntimeError("TOP_SECRET_TRANSCRIPT")


def test_exception_message_does_not_enter_logs_or_response(
    tmp_path: Path, caplog
) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO)
    app = create_app(
        settings=Settings(temp_dir=tmp_path),
        manager=ready_manager(LeakyFailureBackend()),
        start_loader=False,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/transcribe",
            files={"audio": ("secret.m4a", b"audio", "audio/mp4")},
        )
    assert response.status_code == 500
    assert "TOP_SECRET_TRANSCRIPT" not in response.text
    assert "TOP_SECRET_TRANSCRIPT" not in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_optional_application_token(tmp_path: Path) -> None:
    app = create_app(
        settings=Settings(temp_dir=tmp_path, app_token="correct-horse"),
        manager=ready_manager(),
        start_loader=False,
    )
    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
        assert (
            client.post(
                "/v1/rewrite",
                json={"transcript": "private"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/rewrite",
                json={"transcript": "private"},
                headers={"X-Voice-Diary-Key": "correct-horse"},
            ).status_code
            == 200
        )
