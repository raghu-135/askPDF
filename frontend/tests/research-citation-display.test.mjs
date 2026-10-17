import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildCitationDisplayMap,
  rewriteDocCitationTokens,
} from '../src/lib/research-citation-display.ts';

test('rewriteDocCitationTokens replaces doc ids with readable labels', () => {
  const map = buildCitationDisplayMap([
    { source_id: 'doc:hash:3', file_name: 'zephyr.pdf', page: 3 },
  ]);
  const rewritten = rewriteDocCitationTokens('See doc:hash:3 for sales.', map);
  assert.match(rewritten, /zephyr\.pdf, page 3/);
  assert.match(rewritten, /`doc:hash:3`/);
});

test('rewriteDocCitationTokens normalizes combined pdf page and doc id parentheses', () => {
  const map = buildCitationDisplayMap([
    { source_id: 'doc:abc:c-0', file_name: 'zephyr.pdf', page: 1 },
  ]);
  const rewritten = rewriteDocCitationTokens(
    'Revenue grew (zephyr.pdf, page 1 doc:abc:c-0).',
    map,
  );
  assert.match(rewritten, /zephyr\.pdf, page 1/);
  assert.match(rewritten, /`doc:abc:c-0`/);
  assert.doesNotMatch(rewritten, /\(zephyr\.pdf, page 1 doc:/);
});
