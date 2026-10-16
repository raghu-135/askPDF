from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from app.agent.evidence_contract import (
    RECORD_IDENTIFIER_PATTERN,
    SOURCE_IDENTIFIER_PATTERN,
    normalized_source_url,
)

CANVAS_SPEC_VERSION = 1
RESEARCH_CANVAS_ARTIFACT_KIND = "research_canvas"
RESEARCH_CANVAS_MEDIA_TYPE = "application/vnd.askpdf.canvas+json"
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
MAX_QUOTE_TEXT_CHARS = 1200
MAX_QUOTE_LABEL_CHARS = 160
MAX_TIMELINE_EVENTS = 20
MAX_CLAIM_TEXT_CHARS = 600
EvidenceCellTone = Literal["supported", "contradicted", "partial", "absent"]
DagEdgeRelation = Literal["supports", "contradicts", "precedes"]
ClaimPairRelation = Literal["agrees", "contradicts", "extends"]


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


class TableCell(CanvasModel):
    text: str = Field(max_length=MAX_TABLE_CELL_CHARS)
    tone: Optional[EvidenceCellTone] = None


def _coerce_table_cell(cell: Any) -> TableCell:
    if isinstance(cell, TableCell):
        return cell
    if isinstance(cell, str):
        return TableCell(text=cell)
    if isinstance(cell, dict):
        return TableCell.model_validate(cell)
    raise ValueError("table cells must be strings or {text, tone} objects")


class TableBlock(CanvasModel):
    type: Literal["table"] = "table"
    caption: Optional[str] = Field(default=None, max_length=160)
    headers: list[str] = Field(min_length=1, max_length=MAX_TABLE_HEADERS)
    rows: list[list[Union[str, TableCell]]] = Field(min_length=1, max_length=MAX_TABLE_ROWS)

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > MAX_TABLE_CELL_CHARS for item in cleaned):
            raise ValueError("table headers must be non-empty and within the cell limit")
        return cleaned

    @field_validator("rows", mode="after")
    @classmethod
    def coerce_rows(cls, value: list[list[Union[str, TableCell]]]) -> list[list[TableCell]]:
        coerced = [[_coerce_table_cell(cell) for cell in row] for row in value]
        for row in coerced:
            if any(len(cell.text) > MAX_TABLE_CELL_CHARS for cell in row):
                raise ValueError("table cells exceed the character limit")
        return coerced

    @model_validator(mode="after")
    def validate_row_width(self) -> "TableBlock":
        width = len(self.headers)
        for row in self.rows:
            if len(row) != width:
                raise ValueError("every table row must match the header count")
        return self

    @field_serializer("rows")
    def serialize_rows(self, rows: list[list[TableCell]]) -> list[list[Any]]:
        serialized: list[list[Any]] = []
        for row in rows:
            serialized.append([
                cell.text if cell.tone is None else {"text": cell.text, "tone": cell.tone}
                for cell in row
            ])
        return serialized


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


class QuoteBlock(CanvasModel):
    type: Literal["quote"] = "quote"
    file_hash: str = Field(min_length=1)
    sentence_id: int = Field(ge=0)
    label: Optional[str] = Field(default=None, max_length=MAX_QUOTE_LABEL_CHARS)
    text: Optional[str] = Field(default=None, max_length=MAX_QUOTE_TEXT_CHARS)
    page: Optional[int] = Field(default=None, ge=1)

    @field_validator("file_hash")
    @classmethod
    def validate_file_hash(cls, value: str) -> str:
        if not SOURCE_IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError("quote requires a valid file_hash")
        return value


class TimelineEvent(CanvasModel):
    time: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=200)
    note: Optional[str] = Field(default=None, max_length=400)
    citation: CanvasCitation


class TimelineBlock(CanvasModel):
    type: Literal["timeline"] = "timeline"
    title: Optional[str] = Field(default=None, max_length=120)
    events: list[TimelineEvent] = Field(min_length=1, max_length=MAX_TIMELINE_EVENTS)


class ClaimSide(CanvasModel):
    text: str = Field(min_length=1, max_length=MAX_CLAIM_TEXT_CHARS)
    citation: CanvasCitation


class ClaimPairBlock(CanvasModel):
    type: Literal["claim_pair"] = "claim_pair"
    relation: ClaimPairRelation
    left: ClaimSide
    right: ClaimSide


class DagNode(CanvasModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    label: str = Field(min_length=1, max_length=48)


class DagEdge(CanvasModel):
    source: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=64)
    relation: Optional[DagEdgeRelation] = None


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
    Union[
        StatBlock,
        TableBlock,
        CalloutBlock,
        MarkdownBlock,
        SourceListBlock,
        DagBlock,
        QuoteBlock,
        TimelineBlock,
        ClaimPairBlock,
    ],
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


class CanvasAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def iter_canvas_citations(spec: CanvasSpec) -> list[CanvasCitation]:
    citations: list[CanvasCitation] = []
    for section in spec.sections:
        for block in section.blocks:
            if isinstance(block, SourceListBlock):
                citations.extend(block.citations)
            elif isinstance(block, TimelineBlock):
                citations.extend(event.citation for event in block.events)
            elif isinstance(block, ClaimPairBlock):
                citations.append(block.left.citation)
                citations.append(block.right.citation)
    return citations


def iter_canvas_quote_file_hashes(spec: CanvasSpec) -> list[str]:
    hashes: list[str] = []
    for section in spec.sections:
        for block in section.blocks:
            if isinstance(block, QuoteBlock):
                hashes.append(block.file_hash)
    return hashes


def admit_canvas_spec(spec: CanvasSpec, *, thread_file_hashes: set[str]) -> None:
    citations = iter_canvas_citations(spec)
    quote_hashes = iter_canvas_quote_file_hashes(spec)
    if not citations and not quote_hashes:
        raise CanvasAdmissionError(
            "missing_citations",
            "publish_canvas requires at least one citation or quote block",
        )
    unknown = sorted(
        {
            citation.file_hash
            for citation in citations
            if citation.kind == "document" and citation.file_hash not in thread_file_hashes
        }
        | {file_hash for file_hash in quote_hashes if file_hash not in thread_file_hashes}
    )
    if unknown:
        raise CanvasAdmissionError(
            "unknown_document_citation",
            "document citations must use file_hash values attached to this thread: " + ", ".join(unknown),
        )


def _sentence_lookup(parsed: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not parsed:
        return {}
    sentences = parsed.get("sentences") or []
    lookup: dict[int, dict[str, Any]] = {}
    for item in sentences:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        try:
            sentence_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        lookup[sentence_id] = item
    return lookup


def resolve_quote_blocks(
    spec: CanvasSpec,
    *,
    sentences_by_file: dict[str, dict[str, Any] | None],
) -> CanvasSpec:
    """Fill canonical quote text and page from parsed sentences; reject unknown ids."""

    sections: list[CanvasSection] = []
    for section in spec.sections:
        blocks: list[CanvasBlock] = []
        for block in section.blocks:
            if not isinstance(block, QuoteBlock):
                blocks.append(block)
                continue
            lookup = _sentence_lookup(sentences_by_file.get(block.file_hash))
            sentence = lookup.get(block.sentence_id)
            if sentence is None:
                raise CanvasAdmissionError(
                    "unknown_sentence_id",
                    f"quote sentence_id {block.sentence_id} was not found for file {block.file_hash}",
                )
            text = str(sentence.get("text") or "").strip()
            if not text:
                raise CanvasAdmissionError(
                    "empty_quote",
                    f"quote sentence_id {block.sentence_id} has no text for file {block.file_hash}",
                )
            page_raw = sentence.get("page")
            page = int(page_raw) if page_raw is not None else None
            blocks.append(
                block.model_copy(
                    update={
                        "text": text[:MAX_QUOTE_TEXT_CHARS],
                        "page": page if page and page >= 1 else None,
                    }
                )
            )
        sections.append(section.model_copy(update={"blocks": blocks}))
    return spec.model_copy(update={"sections": sections})
