import assert from 'node:assert/strict';
import test from 'node:test';

import {
  REVIEW_DECISION_PANEL_GRID_ROWS,
  REVIEW_LIMITATIONS_MAX_HEIGHT,
  reviewPanelFlexWeights,
  reviewPanelLayoutFitsViewport,
} from '../src/lib/deep-research-review-panel-layout.ts';

test('review panel grid reserves a sticky footer row', () => {
  assert.match(REVIEW_DECISION_PANEL_GRID_ROWS, /^auto auto auto minmax\(0,\s*1fr\) auto$/);
});

test('review panel flex weights keep transcript reachable at 1024x768', () => {
  const weights = reviewPanelFlexWeights(0.58);
  assert.ok(weights.transcript >= 0.12);
  assert.ok(weights.decision > 0);
  assert.equal(Number((weights.transcript + weights.decision).toFixed(2)), 1);
  assert.equal(reviewPanelLayoutFitsViewport(weights, 768), true);
});

test('limitations region uses a bounded max height', () => {
  assert.match(REVIEW_LIMITATIONS_MAX_HEIGHT, /vh/);
});
