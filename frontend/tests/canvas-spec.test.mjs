import assert from 'node:assert/strict';
import test from 'node:test';

import { documentCitationTarget, layoutCanvasDag, canvasSpecCounts } from '../src/lib/canvas-spec.ts';

test('chunk citations can highlight every sentence id', () => {
  assert.deepEqual(
    documentCitationTarget({
      kind: 'document',
      label: 'Chunk',
      file_hash: 'abc',
      sentence_id: 7,
      sentence_ids: [7, 8, 9],
    }),
    { fileHash: 'abc', sentenceId: 7, sentenceIds: [7, 8, 9], play: false },
  );
});

test('document citations expose file hash and sentence id', () => {
  assert.deepEqual(
    documentCitationTarget({ kind: 'document', label: 'Paper', file_hash: 'abc', sentence_id: 4 }),
    { fileHash: 'abc', sentenceId: 4, sentenceIds: [4], play: false },
  );
  assert.equal(documentCitationTarget({ kind: 'web', label: 'Note', url: 'https://example.com' }), null);
});

test('dag layout places sources before dependents', () => {
  const layout = layoutCanvasDag(
    [
      { id: 'claim', label: 'Claim' },
      { id: 'source', label: 'Source' },
    ],
    [{ source: 'claim', target: 'source', relation: 'supports' }],
  );
  const claim = layout.nodes.find((node) => node.id === 'claim');
  const source = layout.nodes.find((node) => node.id === 'source');
  assert.ok(claim && source);
  assert.ok(claim.x < source.x);
  assert.equal(layout.edges.length, 1);
  assert.equal(layout.edges[0].relation, 'supports');
});

test('canvas spec counts sections, stats, and citations', () => {
  const counts = canvasSpecCounts({
    schema_version: 1,
    title: 'Demo',
    sections: [
      {
        title: 'A',
        blocks: [
          { type: 'stat', value: '1', label: 'One' },
          { type: 'sources', citations: [{ kind: 'web', label: 'Note', url: 'https://example.com' }] },
        ],
      },
    ],
  });
  assert.deepEqual(counts, { sections: 1, stats: 1, sources: 1 });
});
