"""Add shared canonical document and hierarchical projection storage.

Revision ID: f7c9e1a2b3d4
Revises: f5b8d3e0c7a2
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f7c9e1a2b3d4"
down_revision = "f5b8d3e0c7a2"
branch_labels = None
depends_on = None


def _jsonb(name, *, nullable=False, default="'{}'::jsonb"):
    return sa.Column(
        name,
        postgresql.JSONB(astext_type=sa.Text()),
        nullable=nullable,
        server_default=sa.text(default) if default is not None else None,
    )


def upgrade() -> None:
    op.create_table(
        "canonical_documents",
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("generation", sa.String(), nullable=False),
        sa.Column("extraction_fingerprint", sa.String(), nullable=False),
        sa.Column("docling_version", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        _jsonb("document_json"),
        _jsonb("source_metadata_json"),
        _jsonb("failure_json", nullable=True, default=None),
        sa.Column("claim_token", sa.String(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("file_hash"),
        sa.CheckConstraint("status in ('pending', 'running', 'completed', 'failed')", name="ck_canonical_documents_status"),
        sa.CheckConstraint("length(btrim(generation)) > 0", name="ck_canonical_documents_generation"),
        sa.CheckConstraint("length(btrim(extraction_fingerprint)) > 0", name="ck_canonical_documents_fingerprint"),
    )
    op.create_index("idx_canonical_documents_generation", "canonical_documents", ["generation"])
    op.create_index("idx_canonical_documents_fingerprint", "canonical_documents", ["extraction_fingerprint"])
    op.create_index("idx_canonical_documents_status", "canonical_documents", ["status"])

    op.create_table(
        "document_processing_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("job_kind", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False, server_default=""),
        sa.Column("generation", sa.String(), nullable=False),
        sa.Column("extraction_fingerprint", sa.String(), nullable=False),
        sa.Column("chunking_fingerprint", sa.String(), nullable=False, server_default=""),
        sa.Column("claim_token", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_hash", "job_kind", "embedding_model", "generation", "extraction_fingerprint", "chunking_fingerprint", name="uq_document_processing_job_target"),
        sa.CheckConstraint("job_kind in ('conversion', 'projection')", name="ck_document_processing_job_kind"),
        sa.CheckConstraint("status in ('pending', 'running', 'completed', 'failed')", name="ck_document_processing_job_status"),
        sa.CheckConstraint("attempts >= 0", name="ck_document_processing_job_attempts"),
        sa.CheckConstraint("length(btrim(file_hash)) > 0", name="ck_document_processing_job_file_hash"),
        sa.CheckConstraint("length(btrim(generation)) > 0", name="ck_document_processing_job_generation"),
    )
    op.create_index("idx_document_processing_job_file", "document_processing_jobs", ["file_hash", "job_kind"])
    op.create_index("idx_document_processing_job_claim", "document_processing_jobs", ["status", "available_at"])

    op.create_table(
        "document_sections",
        sa.Column("section_id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("generation", sa.String(), nullable=False),
        sa.Column("parent_section_id", sa.String(), nullable=True),
        sa.Column("section_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("title", sa.String(), nullable=False, server_default=""),
        _jsonb("heading_path", default="'[]'::jsonb"),
        _jsonb("element_ids", default="'[]'::jsonb"),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("section_id"),
        sa.UniqueConstraint("file_hash", "generation", "section_order", name="uq_document_sections_order"),
    )
    op.create_index("idx_document_sections_file_generation", "document_sections", ["file_hash", "generation"])
    op.create_index("idx_document_sections_parent", "document_sections", ["file_hash", "generation", "parent_section_id"])

    op.create_table(
        "document_elements",
        sa.Column("element_id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("generation", sa.String(), nullable=False),
        sa.Column("element_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("element_type", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("section_id", sa.String(), nullable=True),
        sa.Column("parent_element_id", sa.String(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        _jsonb("provenance_json"),
        _jsonb("structure_json"),
        sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("element_id"),
    )
    op.create_index("idx_document_elements_generation_order", "document_elements", ["file_hash", "generation", "element_order"])
    op.create_index("idx_document_elements_section_order", "document_elements", ["section_id", "element_order"])

    op.create_table(
        "document_chunk_manifests",
        sa.Column("manifest_id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("generation", sa.String(), nullable=False),
        sa.Column("chunking_fingerprint", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("vector_status", sa.String(), nullable=False, server_default="missing"),
        sa.Column("vector_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("expected_chunk_count", sa.Integer(), nullable=False, server_default="0"),
        _jsonb("expected_chunk_ids", default="'[]'::jsonb"),
        _jsonb("expected_source_ids", default="'[]'::jsonb"),
        _jsonb("failure_json", nullable=True, default=None),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("manifest_id"),
        sa.CheckConstraint("status in ('pending', 'running', 'completed', 'failed')", name="ck_document_chunk_manifests_status"),
        sa.CheckConstraint("vector_status in ('missing', 'running', 'completed', 'failed')", name="ck_document_chunk_manifests_vector_status"),
        sa.CheckConstraint("expected_chunk_count >= 0", name="ck_document_chunk_manifests_count"),
    )
    op.create_index("idx_document_chunk_manifests_ready", "document_chunk_manifests", ["file_hash", "embedding_model", "status"])

    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(), nullable=False),
        sa.Column("manifest_id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("chunk_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("body_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("contextualized_text", sa.Text(), nullable=False, server_default=""),
        _jsonb("sentence_ids", default="'[]'::jsonb"),
        _jsonb("source_element_ids", default="'[]'::jsonb"),
        sa.Column("section_id", sa.String(), nullable=True),
        sa.Column("table_id", sa.String(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        _jsonb("metadata_json"),
        sa.ForeignKeyConstraint(["manifest_id"], ["document_chunk_manifests.manifest_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("manifest_id", "chunk_id"),
        sa.UniqueConstraint("manifest_id", "chunk_order", name="uq_document_chunks_order"),
        sa.UniqueConstraint("manifest_id", "source_id", name="uq_document_chunks_source"),
    )
    op.create_index("idx_document_chunks_file_model_order", "document_chunks", ["file_hash", "embedding_model", "chunk_order"])
    op.create_index("idx_document_chunks_section", "document_chunks", ["file_hash", "section_id", "chunk_order"])
    op.create_index("idx_canonical_documents_claim_token", "canonical_documents", ["claim_token"])
    op.create_index("idx_document_processing_jobs_claim_token", "document_processing_jobs", ["claim_token"])
    op.create_index("idx_document_chunk_manifests_published", "document_chunk_manifests", ["file_hash", "embedding_model", "published_at"])
    op.create_index("idx_document_chunks_source_id", "document_chunks", ["source_id"])


def downgrade() -> None:
    op.drop_index("idx_document_processing_job_claim", table_name="document_processing_jobs")
    op.drop_index("idx_document_processing_job_file", table_name="document_processing_jobs")
    op.drop_table("document_processing_jobs")
    op.drop_index("idx_document_chunks_section", table_name="document_chunks")
    op.drop_index("idx_document_chunks_file_model_order", table_name="document_chunks")
    op.drop_index("idx_document_chunks_source_id", table_name="document_chunks")
    op.drop_index("idx_document_chunk_manifests_published", table_name="document_chunk_manifests")
    op.drop_index("idx_document_processing_jobs_claim_token", table_name="document_processing_jobs")
    op.drop_index("idx_canonical_documents_claim_token", table_name="canonical_documents")
    op.drop_table("document_chunks")
    op.drop_index("idx_document_chunk_manifests_ready", table_name="document_chunk_manifests")
    op.drop_table("document_chunk_manifests")
    op.drop_index("idx_document_elements_section_order", table_name="document_elements")
    op.drop_index("idx_document_elements_generation_order", table_name="document_elements")
    op.drop_table("document_elements")
    op.drop_index("idx_document_sections_parent", table_name="document_sections")
    op.drop_index("idx_document_sections_file_generation", table_name="document_sections")
    op.drop_table("document_sections")
    op.drop_index("idx_canonical_documents_status", table_name="canonical_documents")
    op.drop_index("idx_canonical_documents_fingerprint", table_name="canonical_documents")
    op.drop_index("idx_canonical_documents_generation", table_name="canonical_documents")
    op.drop_table("canonical_documents")
