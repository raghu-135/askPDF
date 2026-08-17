"""Add durable runtime operation idempotency metadata.

The runtime checkpoint database is separate from the application database in
production.  The guarded statements keep normal application migrations safe
while allowing this revision to be applied explicitly to the runtime DB.
"""

import os

from alembic import op


revision = "1e8f3a7c5b2d"
down_revision = "0d7e4a9b2c1f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    runtime_database = os.getenv("ALEMBIC_RUNTIME_DATABASE", "").strip().lower() in {"1", "true", "yes", "on"}
    if not runtime_database:
        return
    op.execute(
        """
        ALTER TABLE runtime_executions
            ADD COLUMN IF NOT EXISTS request_fingerprint text,
            ADD COLUMN IF NOT EXISTS last_operation_id text,
            ADD COLUMN IF NOT EXISTS retry_source_attempt integer;

        CREATE TABLE IF NOT EXISTS runtime_operations (
            run_id text not null references runtime_executions(run_id) on delete cascade,
            operation_id text not null,
            operation text not null,
            request_fingerprint text not null,
            attempt integer not null,
            created_at timestamptz not null default now(),
            primary key (run_id, operation_id)
        );
        """
    )


def downgrade() -> None:
    runtime_database = os.getenv("ALEMBIC_RUNTIME_DATABASE", "").strip().lower() in {"1", "true", "yes", "on"}
    if not runtime_database:
        return
    op.execute("DROP TABLE IF EXISTS runtime_operations")
    op.execute(
        """
        ALTER TABLE runtime_executions
            DROP COLUMN IF EXISTS retry_source_attempt,
            DROP COLUMN IF EXISTS last_operation_id,
            DROP COLUMN IF EXISTS request_fingerprint;
        """
    )
