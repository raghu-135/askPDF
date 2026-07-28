import assert from 'node:assert/strict';
import test from 'node:test';

import { deriveChatSentences } from '../src/lib/chat-sentence-cache.ts';

test('deriveChatSentences reuses cached sentence splits and removes stale entries', () => {
  const cache = new Map();
  const first = deriveChatSentences([
    { id: 'm1', content: '**Hello world.** Second sentence.' },
    { id: 'm2', content: 'Another message.' },
  ], cache);

  assert.deepEqual(first.map((sentence) => sentence.text), [
    'Hello world.',
    'Second sentence.',
    'Another message.',
  ]);
  assert.equal(cache.size, 2);
  const cachedM1 = cache.get('m1');

  const second = deriveChatSentences([
    { id: 'm1', content: '**Hello world.** Second sentence.' },
    { id: 'm3', content: 'Replacement message.' },
  ], cache);

  assert.equal(cache.get('m1'), cachedM1);
  assert.equal(cache.has('m2'), false);
  assert.equal(cache.has('m3'), true);
  assert.deepEqual(second.map((sentence) => sentence.messageIndex), [0, 0, 1]);
});
