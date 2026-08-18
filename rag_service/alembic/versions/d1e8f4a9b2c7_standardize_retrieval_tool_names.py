"""Standardize persisted retrieval tool identifiers.

Revision ID: d1e8f4a9b2c7
Revises: 6d2f8a9b3c1e
Create Date: 2026-08-03 00:00:00.000000

"""
from alembic import op


revision = "d1e8f4a9b2c7"
down_revision = "6d2f8a9b3c1e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.rename_retrieval_identifiers(value jsonb)
        RETURNS jsonb LANGUAGE plpgsql IMMUTABLE AS $$
        BEGIN
            CASE jsonb_typeof(value)
                WHEN 'object' THEN
                    RETURN COALESCE((
                        SELECT jsonb_object_agg(
                            CASE key
                                WHEN 'memory_worker' THEN 'thread_conversation_history_worker'
                                WHEN 'long_term_memory_worker' THEN 'durable_memory_worker'
                                WHEN 'timeline_worker' THEN 'thread_events_worker'
                                WHEN 'deep_memory' THEN 'thread_conversation_history'
                                WHEN 'memory_recall' THEN 'durable_memory'
                                WHEN 'thread_timeline' THEN 'thread_events'
                                ELSE key
                            END,
                            pg_temp.rename_retrieval_identifiers(item)
                            ORDER BY CASE key
                                WHEN 'memory_worker' THEN 0
                                WHEN 'long_term_memory_worker' THEN 0
                                WHEN 'timeline_worker' THEN 0
                                WHEN 'deep_memory' THEN 0
                                WHEN 'memory_recall' THEN 0
                                WHEN 'thread_timeline' THEN 0
                                ELSE 1
                            END
                        )
                        FROM jsonb_each(value) AS entry(key, item)
                    ), '{}'::jsonb);
                WHEN 'array' THEN
                    RETURN COALESCE((
                        SELECT jsonb_agg(pg_temp.rename_retrieval_identifiers(item))
                        FROM jsonb_array_elements(value) AS items(item)
                    ), '[]'::jsonb);
                WHEN 'string' THEN
                    RETURN CASE value #>> '{}'
                        WHEN 'memory_worker' THEN to_jsonb('thread_conversation_history_worker'::text)
                        WHEN 'long_term_memory_worker' THEN to_jsonb('durable_memory_worker'::text)
                        WHEN 'timeline_worker' THEN to_jsonb('thread_events_worker'::text)
                        WHEN 'search_conversation_history' THEN to_jsonb('search_thread_conversation_history'::text)
                        WHEN 'search_long_term_memory' THEN to_jsonb('search_durable_memory'::text)
                        WHEN 'search_thread_timeline' THEN to_jsonb('search_thread_events'::text)
                        WHEN 'deep_memory' THEN to_jsonb('thread_conversation_history'::text)
                        WHEN 'memory_recall' THEN to_jsonb('durable_memory'::text)
                        WHEN 'thread_timeline' THEN to_jsonb('thread_events'::text)
                        WHEN 'memory' THEN to_jsonb('thread_conversation_history'::text)
                        WHEN 'long_term_memory' THEN to_jsonb('durable_memory'::text)
                        WHEN 'timeline' THEN to_jsonb('thread_events'::text)
                        ELSE value
                    END;
                ELSE RETURN value;
            END CASE;
        END;
        $$
        """
    )
    op.execute(
        """
        UPDATE agent_workflows
        SET spec_json = pg_temp.rename_retrieval_identifiers(spec_json)
        WHERE NOT is_builtin
        """
    )
    op.execute(
        """
        UPDATE agent_runs
        SET resolved_spec_json = pg_temp.rename_retrieval_identifiers(resolved_spec_json)
        WHERE status IN ('running', 'awaiting_human')
        """
    )


def downgrade() -> None:
    # This semantic data migration is intentionally not reversible.
    pass
