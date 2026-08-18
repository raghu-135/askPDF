import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_WORKBENCH_LAYOUT,
  normalizeWorkbenchLayout,
  readStoredWorkbenchLayout,
  resizeWorkbenchRatio,
  resolveWorkbenchPlacement,
} from '../src/lib/workbench-layout.ts';
import { readStoredLayoutState, writeStoredLayoutState } from '../src/lib/stored-layout-state.ts';
import {
  DEFAULT_BUILDER_LAYOUT,
  normalizeBuilderLayout,
} from '../src/lib/builder-layout.ts';

test('workbench uses the standard side and bottom ratios', () => {
  assert.equal(DEFAULT_WORKBENCH_LAYOUT.sideRatio, 0.32);
  assert.equal(DEFAULT_WORKBENCH_LAYOUT.bottomRatio, 0.4);
  assert.equal(DEFAULT_WORKBENCH_LAYOUT.visible, true);
});

test('auto placement responds to container width', () => {
  assert.equal(resolveWorkbenchPlacement('auto', 1440), 'right');
  assert.equal(resolveWorkbenchPlacement('auto', 900), 'bottom');
});

test('hard narrow-screen fallback preserves safe bottom docking', () => {
  assert.equal(resolveWorkbenchPlacement('left', 680), 'bottom');
  assert.equal(resolveWorkbenchPlacement('right', 680), 'bottom');
});

test('stored and legacy values are migrated and clamped', () => {
  assert.deepEqual(normalizeWorkbenchLayout({ dockMode: 'hidden', sideRatio: 9, bottomRatio: 0.01 }), {
    placement: 'auto',
    visible: false,
    sideRatio: 0.8,
    bottomRatio: 0.25,
  });
});

test('stored layout is read before falling back to defaults', () => {
  const stored = JSON.stringify({ placement: 'left', visible: false, sideRatio: 0.44, bottomRatio: 0.52 });
  const result = readStoredWorkbenchLayout((key) => key === 'layout-key' ? stored : null, 'layout-key', DEFAULT_WORKBENCH_LAYOUT);
  assert.deepEqual(result, {
    placement: 'left',
    visible: false,
    sideRatio: 0.44,
    bottomRatio: 0.52,
  });
});

test('invalid stored layout safely falls back', () => {
  const result = readStoredWorkbenchLayout(() => '{', 'layout-key', DEFAULT_WORKBENCH_LAYOUT);
  assert.deepEqual(result, DEFAULT_WORKBENCH_LAYOUT);
});

test('pointer ratios honor placement and bounds', () => {
  assert.equal(resizeWorkbenchRatio({ placement: 'right', clientX: 700, clientY: 0, left: 0, right: 1000, top: 0, bottom: 800 }), 0.3);
  assert.equal(resizeWorkbenchRatio({ placement: 'left', clientX: 300, clientY: 0, left: 0, right: 1000, top: 0, bottom: 800 }), 0.3);
  assert.equal(resizeWorkbenchRatio({ placement: 'bottom', clientX: 0, clientY: 480, left: 0, right: 1000, top: 0, bottom: 800 }), 0.4);
});

test('generic stored layout reader normalizes stored and fallback values', () => {
  const stored = JSON.stringify({ placement: 'right', visible: true, sideRatio: 2, bottomRatio: 0.5 });
  const result = readStoredLayoutState((key) => key === 'layout-key' ? stored : null, 'layout-key', DEFAULT_WORKBENCH_LAYOUT, normalizeWorkbenchLayout);
  assert.deepEqual(result, {
    placement: 'right',
    visible: true,
    sideRatio: 0.8,
    bottomRatio: 0.5,
  });
});

test('generic stored layout reader falls back on invalid JSON', () => {
  const result = readStoredLayoutState(() => '{', 'layout-key', DEFAULT_WORKBENCH_LAYOUT, normalizeWorkbenchLayout);
  assert.deepEqual(result, DEFAULT_WORKBENCH_LAYOUT);
});

test('generic stored layout writer serializes state to the requested key', () => {
  const writes = [];
  writeStoredLayoutState((key, value) => writes.push([key, value]), 'layout-key', { visible: false });
  assert.deepEqual(writes, [['layout-key', '{"visible":false}']]);
});

test('builder graph elements layout clamps ratio and preserves collapsed state', () => {
  assert.deepEqual(normalizeBuilderLayout({ graphElementsRatio: 9, graphElementsCollapsed: true }), {
    graphElementsRatio: 0.8,
    graphElementsCollapsed: true,
  });
  assert.deepEqual(normalizeBuilderLayout(null), DEFAULT_BUILDER_LAYOUT);
});
