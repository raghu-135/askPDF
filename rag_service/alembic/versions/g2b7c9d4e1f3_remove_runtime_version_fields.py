"""Remove the persisted application-side runtime binding version."""

from alembic import op


revision = "g2b7c9d4e1f3"
down_revision = "f1a9c7e3b5d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE agent_runs DROP COLUMN IF EXISTS runtime_binding_version")


def downgrade() -> None:
    raise RuntimeError("Runtime version fields were intentionally removed and are not restored")
