import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  adjacentReadableId,
  highlightIdSet,
  readableSentenceIds,
  sentenceById,
} from '../src/lib/sentence-catalog.ts';

test('readable playlist skips non-readable catalog ids', () => {
  const sentences = [
    { id: 0, readable: false },
    { id: 1, readable: true },
    { id: 2, readable: true },
  ];
  assert.deepEqual(readableSentenceIds(sentences), [1, 2]);
  assert.equal(adjacentReadableId(sentences, 1, 1), 2);
  assert.equal(adjacentReadableId(sentences, 2, -1), 1);
});

test('highlight lookup uses cited ids instead of array offsets', () => {
  const sentences = [
    { id: 0, readable: false },
    { id: 5, readable: true },
  ];
  assert.equal(sentenceById(sentences, 5)?.id, 5);
  assert.equal(sentenceById(sentences, 1), undefined);
  assert.deepEqual([...highlightIdSet(null, [2, 5])], [2, 5]);
});
