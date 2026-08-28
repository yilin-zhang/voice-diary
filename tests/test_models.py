from __future__ import annotations

import pytest

from voice_diary.models import ModelOutputError, _json_object


def test_extracts_json_from_code_fence() -> None:
    assert _json_object('```json\n{"diary":"今天很好"}\n```') == {"diary": "今天很好"}


def test_rejects_non_json_editor_output() -> None:
    with pytest.raises(ModelOutputError):
        _json_object("这不是 JSON")
