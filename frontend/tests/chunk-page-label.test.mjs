import assert from 'node:assert/strict';
import test from 'node:test';
import { chunkPageLabel } from '../src/lib/chunk-page-label.ts';

test('chunkPageLabel formats single and ranged pages', () => {
  assert.equal(chunkPageLabel({ page_start: 4 }), 'p. 4');
  assert.equal(chunkPageLabel({ page_start: 31, page_end: 32 }), 'p. 31-32');
});

test('chunkPageLabel falls back to pages string or empty label', () => {
  assert.equal(chunkPageLabel({ pages: '1,2,3' }), 'pages 1,2,3');
  assert.equal(chunkPageLabel({}), '');
});
