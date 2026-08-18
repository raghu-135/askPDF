import assert from 'node:assert/strict';
import { test } from 'node:test';

import { formatDurationMs } from '../src/lib/formatDuration.ts';

test('formatDurationMs adapts millisecond values to readable units', () => {
  assert.equal(formatDurationMs(42), '42ms');
  assert.equal(formatDurationMs(999), '999ms');
  assert.equal(formatDurationMs(1200), '1.2s');
  assert.equal(formatDurationMs(65_000), '1m 5s');
});

test('formatDurationMs handles invalid and zero values explicitly', () => {
  assert.equal(formatDurationMs(undefined), null);
  assert.equal(formatDurationMs(Number.NaN), null);
  assert.equal(formatDurationMs(-1), null);
  assert.equal(formatDurationMs(0), null);
  assert.equal(formatDurationMs(0, { showZero: true }), '0ms');
});
