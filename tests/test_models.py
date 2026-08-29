from __future__ import annotations

import pytest

from voice_diary.models import ModelOutputError, _editor_token_budget, _json_object


def test_extracts_json_from_code_fence() -> None:
    assert _json_object('```json\n{"diary":"今天很好"}\n```') == {"diary": "今天很好"}


def test_rejects_non_json_editor_output() -> None:
    with pytest.raises(ModelOutputError):
        _json_object("这不是 JSON")


def test_editor_token_budget_scales_with_transcript_and_respects_cap() -> None:
    assert _editor_token_budget(20, 8192) == 512
    assert _editor_token_budget(1000, 8192) == 2256
    assert _editor_token_budget(5000, 8192) == 8192
