import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildRecoveredPendingHumanReviewMessages,
  hasPersistedUserPromptForRun,
  resolveHumanReviewConversationRunId,
} from '../src/lib/pending-human-review-recovery.ts';

const parentQuestion = 'Find the paper PDF in Google Scholar for the uploaded document.';
const persistedParentUser = {
  id: 'user-msg-1',
  role: 'user',
  content: parentQuestion,
  agent_run_id: 'parent-auto-run',
  agent_run_turn_kind: 'user_prompt',
};

test('resolveHumanReviewConversationRunId follows parent_run_id for delegated child approvals', () => {
  assert.equal(
    resolveHumanReviewConversationRunId({ id: 'child-browser-run', parent_run_id: 'parent-auto-run' }),
    'parent-auto-run',
  );
  assert.equal(
    resolveHumanReviewConversationRunId({ id: 'parent-auto-run', parent_run_id: null }),
    'parent-auto-run',
  );
});

test('delegated child pending review does not synthesize a second user prompt', () => {
  const existing = [persistedParentUser];
  const recovered = buildRecoveredPendingHumanReviewMessages({
    run: { id: 'child-browser-run', parent_run_id: 'parent-auto-run', started_at: '2026-01-01T00:00:00.000Z' },
    interruptTitle: 'Approve browser tool?',
    recoveredQuestion: parentQuestion,
    localUserMessageId: 'recovered-review-user-child-browser-run',
    localAssistantMessageId: 'recovered-review-asst-child-browser-run',
    existingMessages: existing,
    interruptRequestedAt: '2026-01-01T00:00:01.000Z',
  });
  assert.ok(recovered);
  assert.equal(recovered.length, 1);
  assert.equal(recovered[0].role, 'assistant');
  assert.equal(recovered[0].agent_run_id, 'child-browser-run');
  const userRows = [...existing, ...recovered].filter((row) => row.role === 'user');
  assert.equal(userRows.length, 1);
  assert.equal(hasPersistedUserPromptForRun(userRows, 'parent-auto-run'), true);
});

test('non-delegated awaiting_human still recovers user and assistant rows when messages are empty', () => {
  const recovered = buildRecoveredPendingHumanReviewMessages({
    run: { id: 'run-1', parent_run_id: null, started_at: '2026-01-01T00:00:00.000Z' },
    interruptTitle: 'Approve web search?',
    recoveredQuestion: 'Search the web for the latest version',
    localUserMessageId: 'recovered-review-user-run-1',
    localAssistantMessageId: 'recovered-review-asst-run-1',
    existingMessages: [],
    interruptRequestedAt: '2026-01-01T00:00:01.000Z',
  });
  assert.ok(recovered);
  assert.equal(recovered.length, 2);
  assert.equal(recovered[0].agent_run_id, 'run-1');
  assert.equal(recovered[1].agent_run_id, 'run-1');
});

test('recovery is idempotent when the pending assistant row already exists on the child run', () => {
  const existing = [
    persistedParentUser,
    {
      id: 'recovered-review-asst-child-browser-run',
      role: 'assistant',
      content: 'Approve browser tool?',
      agent_run_id: 'child-browser-run',
      agent_run_turn_kind: 'assistant_pending_review',
      pending_human_review: true,
    },
  ];
  const recovered = buildRecoveredPendingHumanReviewMessages({
    run: { id: 'child-browser-run', parent_run_id: 'parent-auto-run' },
    interruptTitle: 'Approve browser tool?',
    recoveredQuestion: parentQuestion,
    localUserMessageId: 'recovered-review-user-child-browser-run',
    localAssistantMessageId: 'recovered-review-asst-child-browser-run',
    existingMessages: existing,
  });
  assert.equal(recovered, null);
});