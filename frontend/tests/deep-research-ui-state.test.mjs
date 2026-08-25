import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isRunOwnedBySelectedTask,
  isTaskOwnedAgentRun,
  isTerminalAgentTaskEvent,
  mergeActiveAgentTaskRun,
  resolveDeepResearchContextWindow,
  shouldPollAgentTask,
  shouldRefreshAgentTaskTimeline,
  shouldSubscribeToAgentTaskEvents,
} from '../src/lib/deep-research-ui-state.ts';

test('authoritative active-run state and binding replace a stale task run projection', () => {
  const interrupt = { interrupt_id: 'interrupt-1', status: 'pending', allowed_actions: ['approve', 'reject'] };
  const task = { status: 'awaiting_approval', active_run: { id: 'run-1', status: 'awaiting_human', runtime_binding_status: 'active', pending_interrupt: interrupt } };
  const runs = [{ id: 'run-1', status: 'queued', runtime_binding_status: 'unbound', pending_interrupt: null }, { id: 'run-0', status: 'failed' }];

  const merged = mergeActiveAgentTaskRun(task, runs);

  assert.equal(merged[0].status, 'awaiting_human');
  assert.equal(merged[0].pending_interrupt, interrupt);
  assert.equal(merged[0].runtime_binding_status, 'active');
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

test('LangGraph and Hermes stop event subscriptions at the shared terminal boundary', () => {
  for (const engine of ['langgraph', 'hermes']) {
    const activeTask = { status: 'running', configuration: { engine } };
    assert.equal(shouldSubscribeToAgentTaskEvents(activeTask, { status: 'running' }), true, engine);
    assert.equal(shouldSubscribeToAgentTaskEvents({ ...activeTask, status: 'completed' }, { status: 'running' }), false, engine);
    assert.equal(shouldSubscribeToAgentTaskEvents(activeTask, { status: 'completed' }), false, engine);
  }
});

test('committed task terminal events are recognized without framework-specific payloads', () => {
  assert.equal(isTerminalAgentTaskEvent({ type: 'run.completed', terminal: true }), true);
  assert.equal(isTerminalAgentTaskEvent({ type: 'run.failed' }), true);
  assert.equal(isTerminalAgentTaskEvent({ type: 'run.cancelled' }), true);
  assert.equal(isTerminalAgentTaskEvent({ type: 'output.delta', terminal: false }), false);
});

test('terminal task events refresh the persisted timeline even when the run id is unchanged', () => {
  assert.equal(shouldRefreshAgentTaskTimeline({ type: 'task.completed', terminal: true }), true);
  assert.equal(shouldRefreshAgentTaskTimeline({ type: 'run.completed' }), true);
  assert.equal(shouldRefreshAgentTaskTimeline({ type: 'output.delta', terminal: false }), true);
  assert.equal(shouldRefreshAgentTaskTimeline({ type: 'task.running', terminal: false }), false);
});

test('task-owned runs cannot use Debug Trace as an approval surface', () => {
  assert.equal(isTaskOwnedAgentRun({ id: 'run-1', task_id: 'task-1' }), true);
  assert.equal(isTaskOwnedAgentRun({ id: 'run-2', task_id: null }), false);
});

test('task navigation never combines the newly selected task with a stale run', () => {
  assert.equal(isRunOwnedBySelectedTask('task-b', { id: 'run-a', task_id: 'task-a' }), false);
  assert.equal(isRunOwnedBySelectedTask('task-b', { id: 'run-b', task_id: 'task-b' }), true);
  assert.equal(isRunOwnedBySelectedTask(null, { id: 'run-b', task_id: 'task-b' }), false);
});

test('Hermes uses its deployment context without changing LangGraph selection', () => {
  assert.equal(resolveDeepResearchContextWindow('hermes', 20_000, 32_768), 32_768);
  assert.equal(resolveDeepResearchContextWindow('langgraph', 20_000, 32_768), 20_000);
  assert.equal(resolveDeepResearchContextWindow('hermes', 20_000, null), 20_000);
});
