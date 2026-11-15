import assert from 'node:assert/strict';
import test from 'node:test';

import { AgentRunResumeAction } from '../src/lib/enums.ts';
import { toolApprovalTitle } from '../src/lib/tool-approval-presentation.ts';

test('tool approval title copy stays lean for web search interrupts', () => {
  assert.equal(toolApprovalTitle('search_web', {}), 'Allow a web search?');
});

test('tool approval resume actions include once, scope, and skip', () => {
  const wireValues = [
    AgentRunResumeAction.Approve,
    AgentRunResumeAction.ApproveForScope,
    AgentRunResumeAction.ContinueWithout,
  ];
  assert.deepEqual(wireValues, ['approve', 'approve_for_scope', 'continue_without']);
});