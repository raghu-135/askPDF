"""Add durable embedding materialization jobs.

Revision ID: d5f1a2b3c4e6
Revises: c4e8a1b6d2f0
"""

from alembic import op
import sqlalchemy as sa


revision = "d5f1a2b3c4e6"
down_revision = "c4e8a1b6d2f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embedding_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("resource_type", sa.String(), nullable=False),
        sa.Column("resource_id", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("source_version", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("resource_type in ('document', 'chat_memory', 'global_memory')", name="ck_embedding_job_resource_type"),
        sa.CheckConstraint("status in ('pending', 'running', 'completed', 'failed')", name="ck_embedding_job_status"),
        sa.CheckConstraint("attempts >= 0", name="ck_embedding_job_attempts"),
        sa.CheckConstraint("length(btrim(resource_id)) > 0", name="ck_embedding_job_resource_id_nonempty"),
        sa.CheckConstraint("length(btrim(scope_id)) > 0", name="ck_embedding_job_scope_id_nonempty"),
        sa.CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_embedding_job_model_nonempty"),
        sa.CheckConstraint("length(btrim(source_version)) > 0", name="ck_embedding_job_source_version_nonempty"),
        sa.UniqueConstraint("resource_type", "resource_id", "scope_id", "embedding_model", name="uq_embedding_job_target"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_embedding_job_claim", "embedding_jobs", ["status", "available_at"])
    op.create_index("idx_embedding_job_model_status", "embedding_jobs", ["resource_type", "embedding_model", "status"])


def downgrade() -> None:
    op.drop_index("idx_embedding_job_model_status", table_name="embedding_jobs")
    op.drop_index("idx_embedding_job_claim", table_name="embedding_jobs")
    op.drop_table("embedding_jobs")
