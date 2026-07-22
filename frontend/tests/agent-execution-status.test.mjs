import assert from 'node:assert/strict';
import test from 'node:test';

import {
  getAgentExecutionStatusPresentation,
  normalizeAgentExecutionStatus,
} from '../src/components/agent-graph/agent-execution-status.ts';

test('normalizes live, retained, and graph execution statuses', () => {
  assert.equal(normalizeAgentExecutionStatus('completed'), 'completed');
  assert.equal(normalizeAgentExecutionStatus('failed'), 'failed');
  assert.equal(normalizeAgentExecutionStatus('error'), 'failed');
  assert.equal(normalizeAgentExecutionStatus('skipped'), 'skipped');
  assert.equal(normalizeAgentExecutionStatus('running'), 'active');
  assert.equal(normalizeAgentExecutionStatus('active'), 'active');
  assert.equal(normalizeAgentExecutionStatus('awaiting_human'), 'interrupted');
  assert.equal(normalizeAgentExecutionStatus('cancelled'), 'cancelled');
  assert.equal(normalizeAgentExecutionStatus('planned'), 'planned');
  assert.equal(normalizeAgentExecutionStatus(undefined), 'inactive');
});

test('assigns concise accessible status presentations', () => {
  assert.deepEqual(getAgentExecutionStatusPresentation('completed'), {
    kind: 'completed', label: 'Completed', color: 'success', icon: 'check',
  });
  assert.equal(getAgentExecutionStatusPresentation('failed').icon, 'cross');
  assert.equal(getAgentExecutionStatusPresentation('skipped').color, 'warning');
  assert.equal(getAgentExecutionStatusPresentation('active').icon, 'spinner');
  assert.equal(getAgentExecutionStatusPresentation('interrupted').icon, 'pause');
});
