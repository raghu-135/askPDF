"""Remove legacy thread mode and enforce project embedding locks.

Revision ID: c9e6a1b4d3f8
Revises: b8d5f0a3c2e7
Create Date: 2026-07-25 00:00:00.000000
"""

from __future__ import annotations

import uuid

from alembic import op
import sqlalchemy as sa


revision = "c9e6a1b4d3f8"
down_revision = "b8d5f0a3c2e7"
branch_labels = None
depends_on = None


def _foreign_key_names(bind: sa.Connection) -> set[str]:
    return {
        foreign_key["name"]
        for foreign_key in sa.inspect(bind).get_foreign_keys("threads")
        if foreign_key.get("name")
    }


def _index_names(bind: sa.Connection) -> set[str]:
    return {
        index["name"]
        for index in sa.inspect(bind).get_indexes("threads")
        if index.get("name")
    }


def upgrade() -> None:
    bind = op.get_bind()

    # The historical legacy migration installed this trigger to permit thread
    # and project model mismatches. Remove it before normalizing assignments.
    op.execute(
        "drop trigger if exists trg_threads_enforce_project_embedding_model on threads"
    )
    op.execute("drop function if exists enforce_thread_project_embedding_model()")

    mismatched_models = bind.execute(
        sa.text(
            """
            select distinct t.embedding_model
            from threads as t
            join projects as p on p.id = t.project_id
            where t.embedding_model is distinct from p.embedding_model
            order by t.embedding_model
            """
        )
    ).scalars()
    for embedding_model in mismatched_models:
        project_id = bind.scalar(
            sa.text(
                """
                select id
                from projects
                where embedding_model = :embedding_model
                order by created_at, id
                limit 1
                """
            ),
            {"embedding_model": embedding_model},
        )
        if project_id is None:
            project_id = str(uuid.uuid4())
            bind.execute(
                sa.text(
                    """
                    insert into projects (
                        id,
                        name,
                        description,
                        embedding_model,
                        settings_json,
                        created_at
                    )
                    values (
                        :id,
                        :name,
                        'Imported threads grouped by their existing embedding model.',
                        :embedding_model,
                        '{}'::jsonb,
                        now()
                    )
                    """
                ),
                {
                    "id": project_id,
                    "name": f"Imported threads ({embedding_model})",
                    "embedding_model": embedding_model,
                },
            )
        bind.execute(
            sa.text(
                """
                update threads as t
                set project_id = :project_id
                from projects as p
                where p.id = t.project_id
                  and t.embedding_model = :embedding_model
                  and t.embedding_model is distinct from p.embedding_model
                """
            ),
            {
                "project_id": project_id,
                "embedding_model": embedding_model,
            },
        )

    foreign_keys = _foreign_key_names(bind)
    for constraint_name in (
        "fk_threads_project_id_projects",
        "fk_threads_project_embedding_model",
    ):
        if constraint_name in foreign_keys:
            op.drop_constraint(
                constraint_name,
                "threads",
                type_="foreignkey",
            )
    op.create_foreign_key(
        "fk_threads_project_embedding_model",
        "threads",
        "projects",
        ["project_id", "embedding_model"],
        ["id", "embedding_model"],
        ondelete="RESTRICT",
    )

    thread_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("threads")
    }
    if "is_legacy" in thread_columns:
        if "ix_threads_is_legacy" in _index_names(bind):
            op.drop_index("ix_threads_is_legacy", table_name="threads")
        op.drop_column("threads", "is_legacy")


def downgrade() -> None:
    bind = op.get_bind()
    thread_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("threads")
    }
    if "is_legacy" not in thread_columns:
        op.add_column(
            "threads",
            sa.Column(
                "is_legacy",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )
        op.create_index(
            "ix_threads_is_legacy",
            "threads",
            ["is_legacy"],
            unique=False,
        )

    foreign_keys = _foreign_key_names(bind)
    if "fk_threads_project_embedding_model" in foreign_keys:
        op.drop_constraint(
            "fk_threads_project_embedding_model",
            "threads",
            type_="foreignkey",
        )
    if "fk_threads_project_id_projects" not in foreign_keys:
        op.create_foreign_key(
            "fk_threads_project_id_projects",
            "threads",
            "projects",
            ["project_id"],
            ["id"],
            ondelete="RESTRICT",
        )
