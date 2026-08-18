import assert from 'node:assert/strict';
import test from 'node:test';

import {
  canRequestChatCancellation,
  recoverCanceledChat,
} from '../src/lib/chat-run-cancellation.ts';

test('chat cancellation is available only for an identified active run', () => {
  assert.equal(canRequestChatCancellation({ runId: 'run-1', running: true }), true);
  assert.equal(canRequestChatCancellation({ running: true }), false);
  assert.equal(canRequestChatCancellation({ runId: 'run-1', running: false }), false);
  assert.equal(canRequestChatCancellation({ runId: 'run-1', running: true, canceling: true }), false);
});

test('canceled chat recovery removes only optimistic messages and restores the exact question', () => {
  const recovered = recoverCanceledChat(
    [
      { id: 'existing-user', content: 'Earlier question' },
      { id: 'temp-user-1', content: 'Typoed qusetion' },
      { id: 'temp-assistant-1', content: '' },
    ],
    'temp-user-1',
    'temp-assistant-1',
    'Typoed qusetion',
  );

  assert.deepEqual(recovered.messages, [{ id: 'existing-user', content: 'Earlier question' }]);
  assert.equal(recovered.input, 'Typoed qusetion');
});
