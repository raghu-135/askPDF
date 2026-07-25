"""Retain the historical legacy-thread revision.

Revision ID: b8d5f0a3c2e7
Revises: a7c4e9f2b1d6
Create Date: 2026-07-25 00:00:00.000000

Development databases may already be stamped at this revision. New databases
do not need legacy thread mode; the following revision normalizes databases
that previously applied its original implementation.
"""

from __future__ import annotations


revision = "b8d5f0a3c2e7"
down_revision = "a7c4e9f2b1d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
