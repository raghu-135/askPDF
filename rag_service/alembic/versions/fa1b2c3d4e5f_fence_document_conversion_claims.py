"""Fence conversion publication against stale workers."""

from alembic import op
import sqlalchemy as sa


revision = "fa1b2c3d4e5f"
down_revision = "f9e4a7c2d1b0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("canonical_documents", sa.Column("claim_token", sa.String(), nullable=True))
    op.add_column("canonical_documents", sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("document_processing_jobs", sa.Column("claim_token", sa.String(), nullable=True))
    op.create_index("idx_canonical_documents_claim_token", "canonical_documents", ["claim_token"])
    op.create_index("idx_document_processing_jobs_claim_token", "document_processing_jobs", ["claim_token"])


def downgrade() -> None:
    op.drop_index("idx_document_processing_jobs_claim_token", table_name="document_processing_jobs")
    op.drop_index("idx_canonical_documents_claim_token", table_name="canonical_documents")
    op.drop_column("document_processing_jobs", "claim_token")
    op.drop_column("canonical_documents", "claim_token")
    op.drop_column("canonical_documents", "claimed_at")
