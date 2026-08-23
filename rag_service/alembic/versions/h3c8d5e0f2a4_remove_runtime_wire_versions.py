"""Remove obsolete version columns from the runtime-owned store."""

from alembic import op


revision = "h3c8d5e0f2a4"
down_revision = "1e8f3a7c5b2d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runtime_events DROP COLUMN IF EXISTS runtime_version")
    op.execute("ALTER TABLE runtime_events DROP COLUMN IF EXISTS contract_version")
    op.execute("ALTER TABLE runtime_operations ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'queued'")
    op.execute("ALTER TABLE runtime_operations ADD COLUMN IF NOT EXISTS result jsonb")


def downgrade() -> None:
    raise RuntimeError("Runtime wire version fields were intentionally removed")
