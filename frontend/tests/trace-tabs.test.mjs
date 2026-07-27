import assert from 'node:assert/strict';
import test from 'node:test';

import { closeTraceTab, upsertTraceTab } from '../src/lib/trace-tabs.ts';

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

test('closing the active trace selects its left neighbor', () => {
  const result = closeTraceTab([{ id: 'run-1' }, { id: 'run-2' }, { id: 'run-3' }], 'run-3', 'run-3');
  assert.deepEqual(result.tabs.map((tab) => tab.id), ['run-1', 'run-2']);
  assert.equal(result.activeId, 'run-2');
});
