import assert from 'node:assert/strict';
import test from 'node:test';

import { canOpenTraceTab, closeTraceTab, isValidTraceId, pendingTraceTabId, traceTabIdForRun, upsertTraceTab } from '../src/lib/trace-tabs.ts';

test('invalid trace IDs cannot create or activate a trace tab', () => {
  assert.equal(isValidTraceId(undefined), false);
  assert.equal(isValidTraceId(''), false);
  assert.equal(isValidTraceId('temp-assistant-1'), false);
  assert.equal(isValidTraceId('test-assistant-1'), false);
  assert.equal(isValidTraceId('pending-trace:temp-assistant-1'), false);
  assert.equal(isValidTraceId('run-1'), true);
  assert.equal(canOpenTraceTab('pending-trace:temp-assistant-1'), true);
  assert.equal(traceTabIdForRun(undefined, 'temp-assistant-1'), 'pending-trace:temp-assistant-1');
  assert.equal(traceTabIdForRun('run-1', 'temp-assistant-1'), 'run-1');
  assert.deepEqual(upsertTraceTab([{ id: 'run-1' }], { id: undefined }), [{ id: 'run-1' }]);
});

test('opening an existing trace updates it without duplication', () => {
  const tabs = [{ id: 'run-1', label: 'Old', status: 'running' }];
  const next = upsertTraceTab(tabs, { id: 'run-1', label: 'Updated', status: 'completed' });
  assert.equal(next.length, 1);
  assert.deepEqual(next[0], { id: 'run-1', label: 'Updated', status: 'completed' });
});

test('opening another trace preserves multiple runs', () => {
  const next = upsertTraceTab([{ id: 'run-1' }], { id: 'run-2' });
  assert.deepEqual(next.map((tab) => tab.id), ['run-1', 'run-2']);
});

test('a delayed metadata update does not clear a live trace view', () => {
  const liveTraceView = { status: 'running', events: [{ sequence: 1 }] };
  const tabs = [{ id: 'run-1', status: 'running', liveTraceView }];

  const next = upsertTraceTab(tabs, {
    id: 'run-1',
    status: 'running',
    runDetails: { id: 'run-1' },
    liveTraceView: undefined,
  });

  assert.equal(next[0].liveTraceView, liveTraceView);
  assert.deepEqual(next[0].runDetails, { id: 'run-1' });
});

test('closing the active trace selects its left neighbor', () => {
  const result = closeTraceTab([{ id: 'run-1' }, { id: 'run-2' }, { id: 'run-3' }], 'run-3', 'run-3');
  assert.deepEqual(result.tabs.map((tab) => tab.id), ['run-1', 'run-2']);
  assert.equal(result.activeId, 'run-2');
});

test('a pending live tab is replaced by the durable run id', () => {
  const pendingId = pendingTraceTabId('temp-assistant-1');
  const pending = upsertTraceTab([], {
    id: pendingId,
    messageId: 'temp-assistant-1',
    label: 'agent',
    liveTraceView: { status: 'running', events: [{ sequence: 1 }] },
  });
  assert.deepEqual(pending.map((tab) => tab.id), [pendingId]);

  const next = upsertTraceTab(pending, {
    id: 'run-1',
    messageId: 'temp-assistant-1',
    label: 'agent',
    liveTraceView: { status: 'running', events: [{ sequence: 1 }, { sequence: 2 }] },
  });
  assert.deepEqual(next.map((tab) => tab.id), ['run-1']);
  assert.equal(next[0].liveTraceView.events.length, 2);
});
