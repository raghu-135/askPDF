import assert from 'node:assert/strict';
import test from 'node:test';

import { closeTraceTab, isValidTraceId, upsertTraceTab } from '../src/lib/trace-tabs.ts';

test('invalid trace IDs cannot create or activate a trace tab', () => {
  assert.equal(isValidTraceId(undefined), false);
  assert.equal(isValidTraceId(''), false);
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
