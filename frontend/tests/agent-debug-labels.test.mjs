import assert from 'node:assert/strict';
import { test } from 'node:test';

import { formatSkipReason } from '../src/lib/agentDebugLabels.ts';

test('formatSkipReason maps internal skip codes to concise labels', () => {
  assert.equal(formatSkipReason('not_selected_by_plan'), 'Not in plan');
  assert.equal(formatSkipReason('web_search_disabled'), 'Web disabled');
});

test('formatSkipReason falls back to readable unknown codes', () => {
  assert.equal(formatSkipReason('custom_skip_code'), 'custom skip code');
  assert.equal(formatSkipReason(null), null);
});
