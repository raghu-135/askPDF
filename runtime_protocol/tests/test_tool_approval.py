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

    def test_runtime_response_operations_are_explicit(self):
        from runtime_protocol.tool_approval import (
            tool_approval_response_operation, stable_invocation_id,
        )
        self.assertEqual(tool_approval_response_operation("hermes"), "run.approval.respond")
        self.assertEqual(tool_approval_response_operation("langgraph"), "run.resume")
        self.assertEqual(
            stable_invocation_id({"tool": "search_web", "arguments": {"q": 1}}),
            stable_invocation_id({"arguments": {"q": 1}, "tool": "search_web"}),
        )
        self.assertEqual(invocation_digest("tool", {"a": 1, "b": 2}), invocation_digest("tool", {"b": 2, "a": 1}))
        self.assertNotEqual(invocation_digest("tool", {"a": 1}), invocation_digest("tool", {"a": 2}))
        self.assertNotEqual(invocation_digest("tool", {}), invocation_digest("other", {}))
