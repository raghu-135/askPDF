import assert from 'node:assert/strict';
import test from 'node:test';

import { isParsedSentencePayload, transformSentences } from '../src/lib/bbox-derivation.ts';

test('isParsedSentencePayload requires a non-empty array', () => {
  assert.equal(isParsedSentencePayload(undefined), false);
  assert.equal(isParsedSentencePayload(null), false);
  assert.equal(isParsedSentencePayload('{"sentences":[]}'), false);
  assert.equal(isParsedSentencePayload([]), false);
  assert.equal(isParsedSentencePayload([{ id: 1, text: 'Hello', page: 1, bbox: [0, 0, 1, 1], page_width: 100, page_height: 100 }]), true);
});

test('transformSentences treats a missing payload as empty instead of throwing', () => {
  assert.deepEqual(transformSentences(undefined), []);
  assert.deepEqual(transformSentences(null), []);
  const transformed = transformSentences([
    { id: 1, text: 'Hello', label: 'text', page: 1, bbox: [10, 20, 40, 50], page_width: 100, page_height: 200 },
  ]);
  assert.deepEqual(transformed[0].bboxes, [{
    page: 1,
    x: 10,
    y: 20,
    width: 30,
    height: 30,
    page_width: 100,
    page_height: 200,
  }]);
});

test('transformSentences omits placeholder boxes when coordinates are missing', () => {
  const transformed = transformSentences([
    { id: 2, text: 'No coords', label: 'text', page: 1, page_width: 100, page_height: 200 },
  ]);
  assert.deepEqual(transformed[0].bboxes, []);
});

test('deriveBBoxes drops boxes without page dimensions instead of emitting 0x0', () => {
  const transformed = transformSentences([
    {
      id: 3,
      text: 'Missing page size',
      label: 'text',
      page: 1,
      bbox: { page: 1, x0: 10, y0: 20, x1: 40, y1: 50 },
    },
  ]);
  assert.deepEqual(transformed[0].bboxes, []);
});
