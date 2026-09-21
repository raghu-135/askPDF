import assert from 'node:assert/strict';
import test from 'node:test';
import {
  assignDocumentColors,
  chunkGraphLabel,
  documentFill,
  truncateLabel,
} from '../src/components/embeddings/document-colors.ts';

test('assignDocumentColors spaces hues by document count instead of hashing the file hash', () => {
  const hashes = ['aaa', 'bbb', 'ccc'];
  const dark = assignDocumentColors(hashes, 'dark');
  const light = assignDocumentColors(hashes, 'light');

  assert.equal(dark.aaa, documentFill(0, 3, 'dark'));
  assert.equal(dark.bbb, documentFill(1, 3, 'dark'));
  assert.equal(dark.ccc, documentFill(2, 3, 'dark'));
  assert.equal(light.aaa, documentFill(0, 3, 'light'));
  assert.notEqual(dark.aaa, light.aaa);
  assert.notEqual(dark.aaa, dark.bbb);
  assert.notEqual(dark.bbb, dark.ccc);
});

test('assignDocumentColors keeps first-seen document order and skips duplicates', () => {
  const colors = assignDocumentColors(['doc-a', 'doc-b', 'doc-a', 'doc-c'], 'dark');
  assert.deepEqual(Object.keys(colors), ['doc-a', 'doc-b', 'doc-c']);
  assert.equal(colors['doc-a'], documentFill(0, 3, 'dark'));
  assert.equal(colors['doc-c'], documentFill(2, 3, 'dark'));
});

test('assignDocumentColors produces a unique fill for each document as the set grows', () => {
  const hashes = Array.from({ length: 12 }, (_, index) => `doc-${index}`);
  const colors = assignDocumentColors(hashes, 'light');
  const uniqueFills = new Set(Object.values(colors));
  assert.equal(uniqueFills.size, 12);
  assert.match(colors['doc-0'], /^#[0-9a-f]{6}$/);
});

test('chunkGraphLabel shows a short preview of the chunk text', () => {
  assert.equal(
    chunkGraphLabel('  6.4 Cost Efficiency Beyond scores , we examine inference  '),
    '6.4 Cost Efficiency Beyond…',
  );
  assert.equal(chunkGraphLabel(''), '(empty)');
  assert.equal(truncateLabel('short.pdf', 18), 'short.pdf');
});
