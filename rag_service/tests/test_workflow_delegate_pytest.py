"""Tests for Auto delegate catalog resolution."""

from __future__ import annotations

from app.product_orchestration.workflow_delegate import (
    AUTO_WORKFLOW_ID,
    HERMES_WORKFLOW_ID,
    inject_delegate_catalog_into_spec,
    normalized_auto_policy,
)


def test_normalized_auto_policy_excludes_self_and_hermes():
    policy = normalized_auto_policy({"auto_policy": {"delegate_policy": {"exclude_workflow_ids": []}}})
    assert AUTO_WORKFLOW_ID in policy["delegate_policy"]["exclude_workflow_ids"]
    assert HERMES_WORKFLOW_ID in policy["delegate_policy"]["exclude_workflow_ids"]


def test_inject_delegate_catalog_into_spec():
    spec = {"config": {"auto_policy": {}}}
    catalog = [{"workflow_id": "router_rag_agent", "name": "Router"}]
    materialized = inject_delegate_catalog_into_spec(spec, catalog)
    assert materialized["config"]["available_delegate_workflows"] == catalog
