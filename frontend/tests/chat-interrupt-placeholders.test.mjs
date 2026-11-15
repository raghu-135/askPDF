import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clearInterruptPlaceholderForLiveRun,
  isChatInterruptPlaceholderMessage,
  shouldHideChatInterruptPlaceholder,
} from '../src/lib/chat-interrupt-placeholders.ts';
const pendingAssistant = {
  id: 'a-1',
  role: 'assistant',
  content: 'Allow a web search?',
  agent_run_id: 'run-1',
  agent_run_turn_kind: 'assistant_pending_review',
  pending_human_review: true,
};

test('interrupt placeholders stay hidden until resume reveals the assistant row', () => {
  assert.equal(shouldHideChatInterruptPlaceholder(pendingAssistant), true);
  const live = clearInterruptPlaceholderForLiveRun(pendingAssistant);
  assert.equal(isChatInterruptPlaceholderMessage(live), false);
  assert.equal(shouldHideChatInterruptPlaceholder(live), false);
  assert.equal(live.content, '');
  assert.equal(live.pending_human_review, false);
  assert.equal(live.agent_run_turn_kind, 'assistant_final');
});