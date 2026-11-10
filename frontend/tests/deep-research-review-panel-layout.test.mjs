import assert from 'node:assert/strict';
import test from 'node:test';

import {
  REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT,
  REVIEW_DECISION_PANEL_GRID_ROWS,
  REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX,
  REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS,
  REVIEW_LIMITATIONS_MAX_HEIGHT,
  REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX,
  reviewPanelFlexWeights,
  reviewPanelOpenDecisionSlicePx,
} from '../src/lib/deep-research-review-panel-layout.ts';

test('review panel scroll grid keeps a limitations floor without the footer row', () => {
  assert.equal(
    REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS,
    `auto auto minmax(${REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX}px, auto) minmax(0, 1fr)`,
  );
});

test('non-fill panel grid appends a footer row', () => {
  assert.equal(REVIEW_DECISION_PANEL_GRID_ROWS, `${REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS} auto`);
});

test('review panel flex weights keep transcript reachable at 1024x768', () => {
  const weights = reviewPanelFlexWeights(0.58);
  assert.ok(weights.transcript >= 0.12);
  assert.ok(weights.decision > 0);
  assert.equal(Number((weights.transcript + weights.decision).toFixed(2)), 1);
});

test('review open mode caps transcript height', () => {
  assert.equal(REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX, 72);
});

test('review open mode leaves a positive decision slice under dense workbench chrome', () => {
  const chromeAboveDecisionPx = 520;
  assert.ok(reviewPanelOpenDecisionSlicePx(800, chromeAboveDecisionPx) > 0);
  assert.ok(reviewPanelOpenDecisionSlicePx(650, chromeAboveDecisionPx) > 0);
  assert.ok(reviewPanelOpenDecisionSlicePx(768, chromeAboveDecisionPx) > 0);
});

test('limitations region uses a bounded max height', () => {
  assert.match(REVIEW_LIMITATIONS_MAX_HEIGHT, /vh/);
});

test('fixed review header height is bounded for footer visibility', () => {
  assert.match(REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT, /vh/);
});
