import assert from 'node:assert/strict';
import test from 'node:test';

import { getChatComposerState } from '../src/lib/chat-composer-state.ts';

const readyInput = {
  loading: false,
  llmModel: 'gpt-test',
  isLlmModelValid: true,
  isLlmToolsSupported: true,
  isEmbeddingModelValid: true,
  indexingStatus: 'ready',
  hasInput: false,
};

test('chat composer state follows lock precedence', () => {
  const cases = [
    [{ humanReviewPending: true, loading: true }, 'human_review_pending', true, false],
    [{ loading: true, llmModel: '' }, 'sending', true, true],
    [{ llmModel: '' }, 'no_llm_selected', true, false],
    [{ isLlmModelValid: null, isEmbeddingModelValid: null, indexingStatus: 'checking' }, 'llm_checking', true, true],
    [{ isLlmModelValid: false, isEmbeddingModelValid: null, indexingStatus: 'checking' }, 'llm_unavailable', true, false],
    [{ isLlmToolsSupported: false, isEmbeddingModelValid: null, indexingStatus: 'checking' }, 'llm_tools_unsupported', true, false],
    [{ isEmbeddingModelValid: null, indexingStatus: 'checking' }, 'embedding_checking', true, true],
    [{ isEmbeddingModelValid: false, indexingStatus: 'error' }, 'embedding_unavailable', true, false],
    [{ indexingStatus: 'blocked' }, 'embedding_unavailable', true, false],
    [{ indexingStatus: 'error' }, 'index_error', true, false],
    [{ indexingStatus: 'indexing' }, 'indexing', true, true],
    [{ hasInput: true }, 'ready', false, false],
    [{ humanReviewPending: true }, 'human_review_pending', true, false],
  ];

  for (const [overrides, status, disabled, busy] of cases) {
    const state = getChatComposerState({ ...readyInput, ...overrides });
    assert.equal(state.status, status);
    assert.equal(state.disabled, disabled);
    assert.equal(state.busy, busy);
    assert.equal(typeof state.placeholder, 'string');
    assert.ok(state.placeholder.length > 0);
  }
});

test('ready placeholder includes submit hint only when the composer has text', () => {
  assert.doesNotMatch(
    getChatComposerState({ ...readyInput, hasInput: false }).placeholder,
    /Shift\+Enter/
  );
  assert.match(
    getChatComposerState({ ...readyInput, hasInput: true }).placeholder,
    /Shift\+Enter/
  );
});

test('human review placeholder guides the user to resolve pending approval', () => {
  const state = getChatComposerState({ ...readyInput, humanReviewPending: true });
  assert.equal(state.placeholder, 'Approve or skip to continue');
});

test('clarification pending keeps the composer enabled with a clarification placeholder', () => {
  const state = getChatComposerState({ ...readyInput, clarificationPending: true });
  assert.equal(state.status, 'ready');
  assert.equal(state.disabled, false);
  assert.equal(state.busy, false);
  assert.match(state.placeholder, /Pick an option above/);
});

test('human review wins over clarification pending', () => {
  const state = getChatComposerState({
    ...readyInput,
    humanReviewPending: true,
    clarificationPending: true,
  });
  assert.equal(state.status, 'human_review_pending');
  assert.equal(state.disabled, true);
});
