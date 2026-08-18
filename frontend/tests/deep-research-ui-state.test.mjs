import assert from 'node:assert/strict';
import test from 'node:test';

import { isTaskOwnedAgentRun, mergeActiveAgentTaskRun, shouldPollAgentTask } from '../src/lib/deep-research-ui-state.ts';

test('authoritative active-run interrupt replaces a stale task run projection', () => {
  const interrupt = { interrupt_id: 'interrupt-1', status: 'pending', allowed_actions: ['approve', 'reject'] };
  const task = { status: 'awaiting_approval', active_run: { id: 'run-1', status: 'awaiting_human', pending_interrupt: interrupt } };
  const runs = [{ id: 'run-1', status: 'queued', pending_interrupt: null }, { id: 'run-0', status: 'failed' }];

  const merged = mergeActiveAgentTaskRun(task, runs);

  assert.equal(merged[0].status, 'awaiting_human');
  assert.equal(merged[0].pending_interrupt, interrupt);
  assert.equal(merged[1], runs[1]);
});

test('active tasks continue polling while terminal tasks stop', () => {
  for (const status of ['created', 'queued', 'running', 'pausing', 'paused', 'awaiting_approval', 'cancelling']) {
    assert.equal(shouldPollAgentTask({ status }), true, status);
  }
  for (const status of ['completed', 'failed', 'expired', 'cancelled']) {
    assert.equal(shouldPollAgentTask({ status }), false, status);
  }
});

test('task-owned runs cannot use Debug Trace as an approval surface', () => {
  assert.equal(isTaskOwnedAgentRun({ id: 'run-1', task_id: 'task-1' }), true);
  assert.equal(isTaskOwnedAgentRun({ id: 'run-2', task_id: null }), false);
});
