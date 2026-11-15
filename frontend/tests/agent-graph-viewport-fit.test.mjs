import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GRAPH_VIEWPORT_MIN_PX,
  isGraphViewportReady,
} from '../src/components/agent-graph/agent-graph-viewport-fit.ts';

test('isGraphViewportReady rejects hidden or zero-sized containers', () => {
  assert.equal(isGraphViewportReady(0, 520), false);
  assert.equal(isGraphViewportReady(800, 0), false);
  assert.equal(isGraphViewportReady(GRAPH_VIEWPORT_MIN_PX - 1, 520), false);
});

test('isGraphViewportReady accepts visible debug canvas dimensions', () => {
  assert.equal(isGraphViewportReady(GRAPH_VIEWPORT_MIN_PX, GRAPH_VIEWPORT_MIN_PX), true);
  assert.equal(isGraphViewportReady(640, 460), true);
});