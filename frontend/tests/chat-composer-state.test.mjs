import assert from 'node:assert/strict';
import test from 'node:test';

import { getChatComposerState } from '../src/lib/chat-composer-state.ts';

const readyInput = {
  loading: false,
  llmModel: 'gpt-test',
  isLlmModelValid: true,
  isLlmToolsSupported: true,
  isEmbedModelValid: true,
  indexingStatus: 'ready',
  hasInput: false,
};

test('chat composer state follows lock precedence', () => {
  const cases = [
    [{ loading: true, llmModel: '' }, 'sending', true, true],
    [{ llmModel: '' }, 'no_llm_selected', true, false],
    [{ isLlmModelValid: null, isEmbedModelValid: null, indexingStatus: 'checking' }, 'llm_checking', true, true],
    [{ isLlmModelValid: false, isEmbedModelValid: null, indexingStatus: 'checking' }, 'llm_unavailable', true, false],
    [{ isLlmToolsSupported: false, isEmbedModelValid: null, indexingStatus: 'checking' }, 'llm_tools_unsupported', true, false],
    [{ isEmbedModelValid: null, indexingStatus: 'checking' }, 'embed_checking', true, true],
    [{ isEmbedModelValid: false, indexingStatus: 'error' }, 'embed_unavailable', true, false],
    [{ indexingStatus: 'error' }, 'index_error', true, false],
    [{ indexingStatus: 'indexing' }, 'indexing', true, true],
    [{ hasInput: true }, 'ready', false, false],
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
