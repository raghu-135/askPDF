import unittest

from runtime_protocol.tool_approval import (
    ApprovalMode, ApprovalScope, ToolApprovalPolicy, decision_scope,
    effective_mode, invocation_digest,
)


class ToolApprovalTests(unittest.TestCase):
    def test_once_does_not_grant_task_permission(self):
        policy = ToolApprovalPolicy(ApprovalMode.ASK, ApprovalScope.TASK)
        self.assertEqual(decision_scope("approve", policy), ApprovalScope.INVOCATION)
        self.assertEqual(decision_scope("approve_for_scope", policy), ApprovalScope.TASK)

    def test_denial_overrides_prior_allowance(self):
        policy = ToolApprovalPolicy(ApprovalMode.DENY)
        self.assertEqual(effective_mode(policy, grant="allowed"), ApprovalMode.DENY)

    def test_invalid_decision_fails_closed(self):
        with self.assertRaises(ValueError):
            decision_scope("unexpected", ToolApprovalPolicy(ApprovalMode.ASK))

    def test_digest_binds_exact_arguments_and_is_key_order_independent(self):
        self.assertEqual(invocation_digest("tool", {"a": 1, "b": 2}), invocation_digest("tool", {"b": 2, "a": 1}))
        self.assertNotEqual(invocation_digest("tool", {"a": 1}), invocation_digest("tool", {"a": 2}))
        self.assertNotEqual(invocation_digest("tool", {}), invocation_digest("other", {}))
