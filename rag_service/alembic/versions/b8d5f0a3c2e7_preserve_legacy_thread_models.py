"""Preserve pre-project thread models and disable durable memory for them.

Revision ID: b8d5f0a3c2e7
Revises: a7c4e9f2b1d6
Create Date: 2026-07-25 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "b8d5f0a3c2e7"
down_revision = "a7c4e9f2b1d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {
        column["name"]
        for column in sa.inspect(bind).get_columns("threads")
    }
    if "is_legacy" not in columns:
        op.add_column(
            "threads",
            sa.Column(
                "is_legacy",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )

    constraints = {
        constraint.get("name")
        for constraint in sa.inspect(bind).get_foreign_keys("threads")
    }
    if "fk_threads_project_embedding_model" in constraints:
        op.drop_constraint(
            "fk_threads_project_embedding_model",
            "threads",
            type_="foreignkey",
        )
    if "fk_threads_project_id_projects" not in constraints:
        op.create_foreign_key(
            "fk_threads_project_id_projects",
            "threads",
            "projects",
            ["project_id"],
            ["id"],
            ondelete="RESTRICT",
        )

    # Recover the most frequently recorded non-project model from completed
    # file index metadata. The reconciliation migration did not delete these
    # records, so no document re-embedding is required.
    op.execute(
        """
        with model_counts as (
            select
                tf.thread_id,
                model_name,
                count(*) as uses
            from thread_files as tf
            join files as f on f.file_hash = tf.file_hash
            join threads as t on t.id = tf.thread_id
            join projects as p on p.id = t.project_id
            cross join lateral jsonb_object_keys(
                coalesce(
                    (f.file_status::jsonb)->'indexing_status'->'models',
                    '{}'::jsonb
                )
            ) as model_name
            where model_name <> p.embedding_model
            group by tf.thread_id, model_name
        ),
        ranked_models as (
            select
                thread_id,
                model_name,
                row_number() over (
                    partition by thread_id
                    order by uses desc, model_name
                ) as rank
            from model_counts
        )
        update threads as t
        set embedding_model = ranked_models.model_name,
            is_legacy = true
        from ranked_models
        where ranked_models.thread_id = t.id
          and ranked_models.rank = 1
        """
    )

    # The Personal project was the compatibility container for threads that
    # predate project-owned embedding models. Empty legacy threads have no
    # file metadata from which to recover a different model, but still must
    # remain excluded from durable long-term memory.
    op.execute(
        """
        update threads as t
        set is_legacy = true
        from projects as p
        where p.id = t.project_id
          and p.name = 'Personal'
        """
    )

    op.execute(
        """
        create or replace function enforce_thread_project_embedding_model()
        returns trigger as $$
        declare
            locked_model varchar;
        begin
            if new.is_legacy then
                return new;
            end if;
            select embedding_model into locked_model
            from projects
            where id = new.project_id;
            if locked_model is null then
                raise exception 'thread project does not exist';
            end if;
            if new.embedding_model is distinct from locked_model then
                raise exception 'thread embedding_model must match its project';
            end if;
            return new;
        end;
        $$ language plpgsql
        """
    )
    op.execute(
        "drop trigger if exists trg_threads_enforce_project_embedding_model on threads"
    )
    op.execute(
        """
        create trigger trg_threads_enforce_project_embedding_model
        before insert or update of project_id, embedding_model, is_legacy on threads
        for each row execute function enforce_thread_project_embedding_model()
        """
    )
    op.create_index(
        "ix_threads_is_legacy",
        "threads",
        ["is_legacy"],
        unique=False,
    )


def downgrade() -> None:
    raise RuntimeError("Legacy thread model recovery cannot be downgraded")
