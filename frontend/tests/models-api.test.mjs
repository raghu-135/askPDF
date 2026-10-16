import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clearLlmModelHealthCache,
  loadLlmModelHealth,
  peekLlmModelHealth,
} from '../src/lib/llm-model-health-cache.ts';

test('llm health is loaded once per model id', async () => {
  clearLlmModelHealthCache();
  let calls = 0;
  const loader = async (model) => {
    calls += 1;
    return { ready: true, supportsTools: true, canInvokeTools: true };
  };
  const first = await loadLlmModelHealth('google/gemma-4-31b-it:free', loader);
  const second = await loadLlmModelHealth('google/gemma-4-31b-it:free', loader);
  assert.equal(calls, 1);
  assert.equal(first.ready, true);
  assert.equal(second.supportsTools, true);
  assert.equal(peekLlmModelHealth('google/gemma-4-31b-it:free')?.canInvokeTools, true);
  await loadLlmModelHealth('other-model', loader);
  assert.equal(calls, 2);
  clearLlmModelHealthCache();
});

test('failed llm health loads are not cached', async () => {
  clearLlmModelHealthCache();
  let calls = 0;
  const loader = async () => {
    calls += 1;
    throw new Error('network');
  };
  await assert.rejects(() => loadLlmModelHealth('broken-model', loader));
  await assert.rejects(() => loadLlmModelHealth('broken-model', loader));
  assert.equal(calls, 2);
  assert.equal(peekLlmModelHealth('broken-model'), null);
  clearLlmModelHealthCache();
});
