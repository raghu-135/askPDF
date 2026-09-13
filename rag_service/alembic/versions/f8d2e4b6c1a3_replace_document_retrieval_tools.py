"""Migrate persisted workflow specs to hierarchical document retrieval."""

from alembic import op


revision = "f8d2e4b6c1a3"
down_revision = "f7c9e1a2b3d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.replace_document_retrieval_tools(value jsonb)
        RETURNS jsonb LANGUAGE plpgsql IMMUTABLE AS $$
        BEGIN
            CASE jsonb_typeof(value)
                WHEN 'object' THEN
                    RETURN COALESCE((
                        SELECT jsonb_object_agg(key, pg_temp.replace_document_retrieval_tools(item))
                        FROM jsonb_each(value) AS entry(key, item)
                    ), '{}'::jsonb);
                WHEN 'array' THEN
                    RETURN COALESCE((
                        SELECT jsonb_agg(pg_temp.replace_document_retrieval_tools(item))
                        FROM jsonb_array_elements(value) AS items(item)
                    ), '[]'::jsonb);
                WHEN 'string' THEN
                    RETURN CASE value #>> '{}'
                        WHEN 'search_documents' THEN to_jsonb('search_knowledge'::text)
                        WHEN 'search_document_by_id' THEN to_jsonb('search_knowledge'::text)
                        WHEN 'document_evidence' THEN to_jsonb('document_search_knowledge'::text)
                        WHEN 'focused_document_evidence' THEN to_jsonb('document_search_knowledge'::text)
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
        SET spec_json = jsonb_set(
            pg_temp.replace_document_retrieval_tools(spec_json),
            '{config,allowed_tool_ids}',
            CASE
                WHEN COALESCE(pg_temp.replace_document_retrieval_tools(spec_json) #> '{config,allowed_tool_ids}', '[]'::jsonb) ? 'document_search_knowledge'
                THEN COALESCE(pg_temp.replace_document_retrieval_tools(spec_json) #> '{config,allowed_tool_ids}', '[]'::jsonb) || '["document_inspection", "document_context"]'::jsonb
                ELSE COALESCE(pg_temp.replace_document_retrieval_tools(spec_json) #> '{config,allowed_tool_ids}', '[]'::jsonb)
            END,
            true
        )
        WHERE spec_json::text LIKE ANY (ARRAY['%search_documents%', '%search_document_by_id%', '%document_evidence%', '%focused_document_evidence%'])
        """
    )
    # Frozen run specs are immutable execution inputs. Their checkpoints and
    # pending tool calls must continue using the exact identifiers captured at
    # run start; only reusable workflow definitions are migrated above.


def downgrade() -> None:
    # The migration changes persisted workflow semantics and is intentionally
    # not reversible: restoring the removed contracts would expose dead tools.
    pass
