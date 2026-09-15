import assert from 'node:assert/strict';
import test from 'node:test';

import { documentCitationTarget } from '../src/lib/canvas-spec.ts';

test('document citations expose file hash and sentence id', () => {
  assert.deepEqual(
    documentCitationTarget({ kind: 'document', label: 'Paper', file_hash: 'abc', sentence_id: 4 }),
    { fileHash: 'abc', sentenceId: 4 },
  );
  assert.equal(documentCitationTarget({ kind: 'web', label: 'Note', url: 'https://example.com' }), null);
});
