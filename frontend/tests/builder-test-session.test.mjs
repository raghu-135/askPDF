import assert from 'node:assert/strict';
import test from 'node:test';

import {
  emptyBuilderTestSession,
  transientMessagesForRequest,
  workflowSpecFingerprint,
} from '../src/lib/builder-test-session.ts';

test('temporary builder sessions are empty and scoped to a thread', () => {
  assert.deepEqual(emptyBuilderTestSession('thread-1'), { threadId: 'thread-1', messages: [] });
});

test('spec fingerprints are stable across object key order', () => {
  assert.equal(workflowSpecFingerprint({ b: 2, a: { d: 4, c: 3 } }), workflowSpecFingerprint({ a: { c: 3, d: 4 }, b: 2 }));
});

test('only completed turns are sent as transient context', () => {
  const messages = [
    { id: '1', role: 'user', content: 'Earlier question', createdAt: '', status: 'completed' },
    { id: '2', role: 'assistant', content: 'Earlier answer', createdAt: '', status: 'completed' },
    { id: '3', role: 'assistant', content: 'Waiting', createdAt: '', status: 'sending' },
  ];
  assert.deepEqual(transientMessagesForRequest(messages), [
    { role: 'user', content: 'Earlier question' },
    { role: 'assistant', content: 'Earlier answer' },
  ]);
});
