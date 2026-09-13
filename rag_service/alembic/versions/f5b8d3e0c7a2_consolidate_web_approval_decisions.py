"""Move task web grants into the shared tool permission store.

The tool names are a snapshot of the external-tool catalog at migration time.
Audit events are retained; runtime authorization no longer reads them.
"""

from alembic import op

revision = "f5b8d3e0c7a2"
down_revision = "e4a7c2d9b6f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO tool_approval_decisions
            (id, run_id, interrupt_id, tool_name, invocation_id, argument_hash,
             scope, scope_id, decision, created_at)
        SELECT md5(e.id || ':' || tools.name), e.agent_run_id,
               'migrated-web:' || e.id || ':' || tools.name, tools.name,
               'migrated-web:' || e.id, '', 'task', e.task_id,
               CASE e.payload_json->>'status'
                   WHEN 'allowed_for_task' THEN 'allowed' ELSE 'denied' END,
               e.created_at
        FROM (
            SELECT DISTINCT ON (task_id) * FROM agent_task_events
            WHERE event_type = 'approval.responded'
              AND payload_json->>'status' IN ('allowed_for_task', 'denied_for_task')
            ORDER BY task_id, sequence DESC
        ) e
        JOIN agent_runs r ON r.id = e.agent_run_id AND r.task_id = e.task_id
        CROSS JOIN (VALUES ('search_web'), ('wikipedia'), ('wikidata'), ('arxiv'),
            ('pub_med'), ('pubmed'), ('semanticscholar'), ('semantic_scholar'),
            ('stack_exchange'), ('yahoo_finance_news')) AS tools(name)
        ON CONFLICT (run_id, interrupt_id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM tool_approval_decisions WHERE interrupt_id LIKE 'migrated-web:%'")
