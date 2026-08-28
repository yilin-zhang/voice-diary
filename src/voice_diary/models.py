from __future__ import annotations

import json
import logging
import re
import threading
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .schemas import RewriteResult, Transcript

logger = logging.getLogger("voice_diary.models")


class ModelsNotReady(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    pass


class ModelBackend(Protocol):
    def load(self) -> None: ...

    def transcribe(self, audio_path: Path, language: str | None) -> Transcript: ...

    def rewrite(
        self, transcript: str, *, date: str | None, title_hint: str | None
    ) -> RewriteResult: ...


def _json_object(text: str) -> dict[str, Any]:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ModelOutputError("editor did not return a JSON object")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelOutputError("editor returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("editor JSON must be an object")
    return payload


def _timestamps(value: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in value or []:
        if isinstance(item, dict):
            serialized.append(item)
        elif hasattr(item, "model_dump"):
            serialized.append(item.model_dump())
        elif hasattr(item, "_asdict"):
            serialized.append(dict(item._asdict()))
        elif isinstance(item, (list, tuple)):
            serialized.append({"values": list(item)})
    return serialized


class FakeBackend:
    """Deterministic backend for local API and privacy tests. Never deploy as production."""

    def load(self) -> None:
        return None

    def transcribe(self, audio_path: Path, language: str | None) -> Transcript:
        size = audio_path.stat().st_size
        return Transcript(text=f"嗯，测试录音，共 {size} 字节。", language=language or "Chinese")

    def rewrite(
        self, transcript: str, *, date: str | None, title_hint: str | None
    ) -> RewriteResult:
        cleaned = transcript.replace("嗯，", "").replace("嗯", "").strip()
        heading = title_hint or date or "今天"
        return RewriteResult(cleaned_transcript=cleaned, diary=f"# {heading}\n\n{cleaned}")


class QwenBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._asr: Any = None
        self._processor: Any = None
        self._editor: Any = None

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        asr_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
            "max_inference_batch_size": 1,
            "max_new_tokens": 4096,
        }
        if self.settings.aligner_model:
            asr_kwargs.update(
                forced_aligner=self.settings.aligner_model,
                forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
            )
        self._asr = Qwen3ASRModel.from_pretrained(self.settings.asr_model, **asr_kwargs)

        self._processor = AutoProcessor.from_pretrained(self.settings.editor_model)
        self._editor = AutoModelForMultimodalLM.from_pretrained(
            self.settings.editor_model,
            dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._editor.eval()

    def transcribe(self, audio_path: Path, language: str | None) -> Transcript:
        result = self._asr.transcribe(
            audio=str(audio_path),
            language=language or None,
            return_time_stamps=bool(self.settings.aligner_model),
        )[0]
        return Transcript(
            text=result.text,
            language=getattr(result, "language", language),
            timestamps=_timestamps(getattr(result, "time_stamps", None)),
        )

    def rewrite(
        self, transcript: str, *, date: str | None, title_hint: str | None
    ) -> RewriteResult:
        prompt = files("voice_diary.prompts").joinpath("diary_zh.txt").read_text("utf-8")
        context = {
            "date": date,
            "title_hint": title_hint,
            "transcript": transcript,
        }
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(context, ensure_ascii=False),
                    }
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._editor.device)
        output = self._editor.generate(
            **inputs,
            max_new_tokens=self.settings.max_editor_tokens,
            do_sample=False,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self._processor.decode(generated, skip_special_tokens=True)
        payload = _json_object(text)
        try:
            return RewriteResult.model_validate(payload)
        except Exception as exc:
            raise ModelOutputError("editor JSON has the wrong schema") from exc


class ModelManager:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.state = "loading"
        self.error_type: str | None = None
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        try:
            self.backend.load()
        except Exception as exc:
            self.error_type = type(exc).__name__
            self.state = "failed"
            logger.error("model_load_failed error_type=%s", self.error_type)
            return
        self.state = "ready"
        logger.info("models_ready")

    def ensure_ready(self) -> None:
        if self.state != "ready":
            raise ModelsNotReady(self.state)

    def transcribe(self, audio_path: Path, language: str | None) -> Transcript:
        self.ensure_ready()
        with self._inference_lock:
            return self.backend.transcribe(audio_path, language)

    def rewrite(
        self, transcript: str, *, date: str | None, title_hint: str | None
    ) -> RewriteResult:
        self.ensure_ready()
        with self._inference_lock:
            return self.backend.rewrite(transcript, date=date, title_hint=title_hint)
