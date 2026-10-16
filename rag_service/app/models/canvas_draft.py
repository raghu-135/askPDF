from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.canvas import (
    MAX_CANVAS_SUMMARY_CHARS,
    MAX_CANVAS_TITLE_CHARS,
    MAX_CLAIM_TEXT_CHARS,
    EvidenceCellTone,
)

CanvasLayout = Literal["compare", "matrix", "timeline"]


class CanvasDraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_CLAIM_TEXT_CHARS)
    source_ids: list[str] = Field(min_length=1, max_length=8)
    time: Optional[str] = Field(default=None, max_length=80)
    tone: Optional[EvidenceCellTone] = None
    group: Optional[str] = Field(default=None, max_length=120)

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("each claim needs at least one source_id")
        return cleaned


class CanvasPublishDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layout: CanvasLayout
    title: str = Field(min_length=1, max_length=MAX_CANVAS_TITLE_CHARS)
    summary: Optional[str] = Field(default=None, max_length=MAX_CANVAS_SUMMARY_CHARS)
    claims: list[CanvasDraftClaim] = Field(min_length=1, max_length=30)
    supersedes_id: Optional[str] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_timeline_times(self) -> "CanvasPublishDraft":
        if self.layout != "timeline":
            return self
        missing = [claim.text[:40] for claim in self.claims if not (claim.time and claim.time.strip())]
        if missing:
            raise ValueError("timeline layout requires time on every claim")
        return self


class CanvasCompileError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)
