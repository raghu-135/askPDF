"""Add validated retrieval attributes to durable memories.

Revision ID: 7f3c1a9d5e2b
Revises: d1e8f4a9b2c7
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "7f3c1a9d5e2b"
down_revision = "d1e8f4a9b2c7"
branch_labels = None
depends_on = None


DEFAULT_ATTRIBUTES = """'{"kind":"fact","applicability":["task_specific"],"durability":"stable"}'::jsonb"""


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "attributes_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text(DEFAULT_ATTRIBUTES),
        ),
    )
    op.alter_column("memories", "attributes_json", server_default=None)
    op.create_check_constraint(
        "ck_memories_attributes_json",
        "memories",
        "jsonb_typeof(attributes_json) = 'object' "
        "and attributes_json ->> 'kind' in ('preference', 'profile', 'instruction', 'constraint', 'decision', 'fact') "
        "and jsonb_typeof(attributes_json -> 'applicability') = 'array' "
        "and jsonb_array_length(attributes_json -> 'applicability') > 0 "
        "and (attributes_json -> 'applicability') <@ '[\"all_answers\",\"writing\",\"code\",\"research\",\"project\",\"task_specific\"]'::jsonb "
        "and attributes_json ->> 'durability' in ('stable', 'time_sensitive')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_memories_attributes_json", "memories", type_="check")
    op.drop_column("memories", "attributes_json")
