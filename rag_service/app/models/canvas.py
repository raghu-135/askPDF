from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.evidence_contract import (
    RECORD_IDENTIFIER_PATTERN,
    SOURCE_IDENTIFIER_PATTERN,
    normalized_source_url,
)

CANVAS_SPEC_VERSION = 1
MAX_CANVAS_TITLE_CHARS = 160
MAX_CANVAS_SUMMARY_CHARS = 400
MAX_CANVAS_SECTIONS = 12
MAX_SECTION_BLOCKS = 16
MAX_STAT_VALUE_CHARS = 80
MAX_STAT_LABEL_CHARS = 120
MAX_TABLE_HEADERS = 8
MAX_TABLE_ROWS = 30
MAX_TABLE_CELL_CHARS = 400
MAX_MARKDOWN_CHARS = 8000
MAX_CALLOUT_CHARS = 1200
MAX_DAG_NODES = 24
MAX_CITATIONS = 20
MAX_CITATION_LABEL_CHARS = 160


class CanvasModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanvasCitation(CanvasModel):
    kind: Literal["document", "web", "memory", "conversation"]
    label: str = Field(min_length=1, max_length=MAX_CITATION_LABEL_CHARS)
    file_hash: Optional[str] = None
    sentence_id: Optional[int] = Field(default=None, ge=0)
    url: Optional[str] = None
    memory_id: Optional[str] = None
    message_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_locator(self) -> "CanvasCitation":
        if self.kind == "document":
            if not self.file_hash or not SOURCE_IDENTIFIER_PATTERN.fullmatch(self.file_hash):
                raise ValueError("document citations require a valid file_hash")
            return self
        if self.kind == "web":
            url = normalized_source_url(self.url)
            if not url:
                raise ValueError("web citations require an http(s) url")
            self.url = url
            return self
        if self.kind == "memory":
            if not self.memory_id or not RECORD_IDENTIFIER_PATTERN.fullmatch(self.memory_id):
                raise ValueError("memory citations require a valid memory_id")
            return self
        if not self.message_id or not RECORD_IDENTIFIER_PATTERN.fullmatch(self.message_id):
            raise ValueError("conversation citations require a valid message_id")
        return self


class StatBlock(CanvasModel):
    type: Literal["stat"] = "stat"
    value: str = Field(min_length=1, max_length=MAX_STAT_VALUE_CHARS)
    label: str = Field(min_length=1, max_length=MAX_STAT_LABEL_CHARS)
    tone: Literal["neutral", "info", "warning"] = "neutral"


class TableBlock(CanvasModel):
    type: Literal["table"] = "table"
    caption: Optional[str] = Field(default=None, max_length=160)
    headers: list[str] = Field(min_length=1, max_length=MAX_TABLE_HEADERS)
    rows: list[list[str]] = Field(min_length=1, max_length=MAX_TABLE_ROWS)

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > MAX_TABLE_CELL_CHARS for item in cleaned):
            raise ValueError("table headers must be non-empty and within the cell limit")
        return cleaned

    @field_validator("rows")
    @classmethod
    def validate_rows(cls, value: list[list[str]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in value:
            cells = [str(cell) for cell in row]
            if any(len(cell) > MAX_TABLE_CELL_CHARS for cell in cells):
                raise ValueError("table cells exceed the character limit")
            rows.append(cells)
        return rows

    @model_validator(mode="after")
    def validate_row_width(self) -> "TableBlock":
        width = len(self.headers)
        for row in self.rows:
            if len(row) != width:
                raise ValueError("every table row must match the header count")
        return self


class CalloutBlock(CanvasModel):
    type: Literal["callout"] = "callout"
    tone: Literal["info", "warning"] = "info"
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=MAX_CALLOUT_CHARS)


class MarkdownBlock(CanvasModel):
    type: Literal["markdown"] = "markdown"
    text: str = Field(min_length=1, max_length=MAX_MARKDOWN_CHARS)


class SourceListBlock(CanvasModel):
    type: Literal["sources"] = "sources"
    title: str = Field(default="Sources", min_length=1, max_length=80)
    citations: list[CanvasCitation] = Field(min_length=1, max_length=MAX_CITATIONS)


class DagNode(CanvasModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    label: str = Field(min_length=1, max_length=48)


class DagEdge(CanvasModel):
    source: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=64)


class DagBlock(CanvasModel):
    type: Literal["dag"] = "dag"
    title: Optional[str] = Field(default=None, max_length=120)
    nodes: list[DagNode] = Field(min_length=1, max_length=MAX_DAG_NODES)
    edges: list[DagEdge] = Field(default_factory=list, max_length=48)

    @model_validator(mode="after")
    def validate_edges(self) -> "DagBlock":
        ids = {node.id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("dag node ids must be unique")
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids:
                raise ValueError("dag edges must reference node ids")
        return self


CanvasBlock = Annotated[
    Union[StatBlock, TableBlock, CalloutBlock, MarkdownBlock, SourceListBlock, DagBlock],
    Field(discriminator="type"),
]


class CanvasSection(CanvasModel):
    title: str = Field(min_length=1, max_length=120)
    blocks: list[CanvasBlock] = Field(min_length=1, max_length=MAX_SECTION_BLOCKS)


class CanvasSpec(CanvasModel):
    schema_version: Literal[1] = CANVAS_SPEC_VERSION
    title: str = Field(min_length=1, max_length=MAX_CANVAS_TITLE_CHARS)
    summary: Optional[str] = Field(default=None, max_length=MAX_CANVAS_SUMMARY_CHARS)
    sections: list[CanvasSection] = Field(min_length=1, max_length=MAX_CANVAS_SECTIONS)


class CanvasCreateRequest(CanvasModel):
    spec: CanvasSpec
    chat_turn_id: Optional[str] = Field(default=None, max_length=64)
    supersedes_id: Optional[str] = Field(default=None, max_length=64)
    idempotency_key: Optional[str] = Field(default=None, min_length=8, max_length=128)


def parse_canvas_spec(value: Any) -> CanvasSpec:
    return CanvasSpec.model_validate(value)
