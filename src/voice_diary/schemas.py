from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Transcript(BaseModel):
    text: str
    language: str | None = None
    timestamps: list[dict[str, Any]] = Field(default_factory=list)


class RewriteRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=500_000)
    date: str | None = Field(default=None, max_length=64)
    title_hint: str | None = Field(default=None, max_length=200)


class RewriteResult(BaseModel):
    cleaned_transcript: str
    diary: str
    uncertainties: list[str] = Field(default_factory=list)


class ProcessResult(BaseModel):
    transcript: Transcript
    rewrite: RewriteResult


class Health(BaseModel):
    status: str
