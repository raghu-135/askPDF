import assert from 'node:assert/strict';
import test from 'node:test';

import {
  closeInstanceTab,
  mergeInstanceTab,
  selectedInstanceTabValue,
  upsertInstanceTab,
} from '../src/lib/instance-tabs.ts';

test('selected instance tab value ignores stale ids', () => {
  assert.equal(selectedInstanceTabValue([{ id: 'a' }, { id: 'b' }], 'missing'), false);
  assert.equal(selectedInstanceTabValue([{ id: 'a' }, { id: 'b' }], 'b'), 'b');
});

test('upsertInstanceTab opens new tabs and merges existing ones', () => {
  const first = upsertInstanceTab([], { id: 'a', label: 'One' });
  assert.deepEqual(first.map((tab) => tab.id), ['a']);

  const merged = upsertInstanceTab(
    first,
    { id: 'a', label: 'Updated' },
    { merge: (left, right) => mergeInstanceTab(left, right) },
  );
  assert.equal(merged[0].label, 'Updated');
  assert.equal(merged.length, 1);

  const second = upsertInstanceTab(merged, { id: 'b', label: 'Two' });
  assert.deepEqual(second.map((tab) => tab.id), ['a', 'b']);
});

test('closeInstanceTab selects the left neighbor when the active tab closes', () => {
  const result = closeInstanceTab(
    [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
    'c',
    'c',
  );
  assert.deepEqual(result.tabs.map((tab) => tab.id), ['a', 'b']);
  assert.equal(result.activeId, 'b');
});

test('closeInstanceTab keeps active id when closing a background tab', () => {
  const result = closeInstanceTab(
    [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
    'b',
    'c',
  );
  assert.deepEqual(result.tabs.map((tab) => tab.id), ['a', 'b']);
  assert.equal(result.activeId, 'b');
});