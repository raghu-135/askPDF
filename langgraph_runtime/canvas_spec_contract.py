"""Shared canvas_spec_v1 field names for tool schemas and retry coaching (no rag_service import)."""

CANVAS_SPEC_V1_FIELD_HINT = (
    "canvas_spec_v1: schema_version=1, title, optional summary, sections (non-empty). "
    "Each section uses title and blocks (not content). "
    "Blocks: stat (value, label), markdown (text), callout (tone, title, body), table, sources (citations), "
    "dag (nodes with id and label; edges with source and target, not from/to; optional relation "
    "supports|contradicts|precedes), quote (file_hash, sentence_id), timeline (events), claim_pair (left, right, relation). "
    "Document citations use kind=document, label, file_hash."
)
