"""Retain the historical project-memory reconciliation revision.

Revision ID: a7c4e9f2b1d6
Revises: f6a1b2c3d4e5
Create Date: 2026-07-25 00:00:00.000000

This revision was applied by development databases before the project-memory
migrations were consolidated. Its ID must remain in the Alembic graph even
though fresh databases no longer require the lossy reconciliation.
"""

from __future__ import annotations


revision = "a7c4e9f2b1d6"
down_revision = "f6a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
