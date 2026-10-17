import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  adjacentReadableId,
  highlightIdSet,
  readableSentenceIds,
  sentenceById,
  uniqueHighlightBoxes,
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

function firstHighlightableBBox(sentences, sentenceIds) {
  if (!sentences?.length || !sentenceIds.length) return null;
  for (const id of sentenceIds) {
    const sentence = sentences.find((item) => item.id === id);
    for (const bbox of sentence?.bboxes ?? []) {
      if (bbox.width > 0 && bbox.height > 0) return bbox;
    }
  }
  return null;
}

test('firstHighlightableBBox skips zero-size placeholders', () => {
  const sentences = [
    { id: 1, bboxes: [{ page: 1, x: 0, y: 0, width: 0, height: 0 }] },
    { id: 2, bboxes: [{ page: 2, x: 10, y: 20, width: 30, height: 8 }] },
  ];
  assert.equal(firstHighlightableBBox(sentences, [1, 2])?.page, 2);
});

test('adjacentReadableId steps from a non-readable id toward readable neighbors', () => {
  const sentences = [
    { id: 0, readable: false },
    { id: 1, readable: true },
    { id: 2, readable: true },
  ];
  assert.equal(adjacentReadableId(sentences, 0, 1), 1);
  assert.equal(adjacentReadableId(sentences, 0, -1), null);
});

test('uniqueHighlightBoxes dedupes identical boxes across sentences', () => {
  const box = { page: 3, x: 10, y: 20, width: 100, height: 12, page_width: 600, page_height: 800 };
  const near = { page: 3, x: 10.2, y: 20.1, width: 100.1, height: 12, page_width: 600, page_height: 800 };
  const distinct = { page: 3, x: 200, y: 20, width: 80, height: 12, page_width: 600, page_height: 800 };
  assert.equal(uniqueHighlightBoxes([box, near, distinct]).length, 2);
});

test('uniqueHighlightBoxes drops line boxes contained in paragraph boxes', () => {
  const paragraph = { page: 5, x: 72, y: 400, width: 420, height: 48, page_width: 600, page_height: 800 };
  const line = { page: 5, x: 74, y: 410, width: 300, height: 14, page_width: 600, page_height: 800 };
  const kept = uniqueHighlightBoxes([line, paragraph]);
  assert.equal(kept.length, 1);
  assert.equal(kept[0].height, 48);
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
