"""Add durable runtime execution lease and fencing columns.

Revision ID: 0d7e4a9b2c1f
Revises: 9b4d6e2f1a7c
Create Date: 2026-08-17 00:00:00.000000

The runtime execution store lives in the runtime checkpoint database, which
is separate from the application database in production.  The guarded SQL
keeps the application Alembic database compatible when this revision is
applied to the normal application database, while allowing the same revision
to be run explicitly against the runtime database.
"""

import os

from alembic import op


revision = "0d7e4a9b2c1f"
down_revision = "9b4d6e2f1a7c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    runtime_database = os.getenv("ALEMBIC_RUNTIME_DATABASE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if runtime_database:
        op.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_executions (
                run_id text primary key,
                operation text not null,
                request jsonb not null,
                payload jsonb not null,
                status text not null,
                cancel_requested boolean not null default false,
                next_sequence integer not null default 1,
                attempt integer not null default 1,
                continuation jsonb,
                result jsonb,
                error jsonb,
                created_at timestamptz not null default now(),
                updated_at timestamptz not null default now()
            );
            """
        )
        op.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_events (
                run_id text not null references runtime_executions(run_id) on delete cascade,
                sequence integer not null,
                attempt integer not null default 1,
                event_id text not null,
                kind text not null,
                payload jsonb not null,
                occurred_at text,
                trace_id text,
                continuation jsonb,
                terminal boolean not null default false,
                result jsonb,
                created_at timestamptz not null default now(),
                primary key (run_id, sequence),
                unique (run_id, event_id)
            );
            """
        )
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.runtime_executions') IS NOT NULL THEN
                ALTER TABLE runtime_executions
                    ADD COLUMN IF NOT EXISTS owner_id text,
                    ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz,
                    ADD COLUMN IF NOT EXISTS heartbeat_at timestamptz,
                    ADD COLUMN IF NOT EXISTS fencing_token bigint NOT NULL DEFAULT 0;
            END IF;

            IF to_regclass('public.runtime_events') IS NOT NULL THEN
                ALTER TABLE runtime_events
                    ADD COLUMN IF NOT EXISTS attempt integer NOT NULL DEFAULT 1,
                    ADD COLUMN IF NOT EXISTS occurred_at text,
                    ADD COLUMN IF NOT EXISTS trace_id text,
                    ADD COLUMN IF NOT EXISTS continuation jsonb;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.runtime_events') IS NOT NULL THEN
                ALTER TABLE runtime_events
                    DROP COLUMN IF EXISTS continuation,
                    DROP COLUMN IF EXISTS trace_id,
                    DROP COLUMN IF EXISTS occurred_at,
                    DROP COLUMN IF EXISTS attempt;
            END IF;

            IF to_regclass('public.runtime_executions') IS NOT NULL THEN
                ALTER TABLE runtime_executions
                    DROP COLUMN IF EXISTS fencing_token,
                    DROP COLUMN IF EXISTS heartbeat_at,
                    DROP COLUMN IF EXISTS lease_expires_at,
                    DROP COLUMN IF EXISTS owner_id;
            END IF;
        END $$;
        """
    )
