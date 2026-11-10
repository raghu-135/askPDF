import assert from 'node:assert/strict';
import test from 'node:test';

import {
  REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT,
  REVIEW_DECISION_PANEL_GRID_ROWS,
  REVIEW_LIMITATIONS_MAX_HEIGHT,
  reviewPanelFlexWeights,
  reviewPanelLayoutFitsViewport,
} from '../src/lib/deep-research-review-panel-layout.ts';

test('review panel grid reserves a sticky footer row with shrinkable header', () => {
  assert.match(
    REVIEW_DECISION_PANEL_GRID_ROWS,
    /^auto auto minmax\(0,\s*auto\) minmax\(0,\s*1fr\) auto$/,
  );
});

test('review panel flex weights keep transcript reachable at 1024x768', () => {
  const weights = reviewPanelFlexWeights(0.58);
  assert.ok(weights.transcript >= 0.12);
  assert.ok(weights.decision > 0);
  assert.equal(Number((weights.transcript + weights.decision).toFixed(2)), 1);
  assert.equal(reviewPanelLayoutFitsViewport(weights, 768), true);
});

test('review panel layout fits workbench chrome at 1280x800 and 1280x650', () => {
  const weights = reviewPanelFlexWeights(0.48);
  assert.equal(reviewPanelLayoutFitsViewport(weights, 800), true);
  assert.equal(reviewPanelLayoutFitsViewport(weights, 650), true);
});

test('limitations region uses a bounded max height', () => {
  assert.match(REVIEW_LIMITATIONS_MAX_HEIGHT, /vh/);
});

test('fixed review header height is bounded for footer visibility', () => {
  assert.match(REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT, /vh/);
});
