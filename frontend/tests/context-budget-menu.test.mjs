import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const TEST_RATIOS = {
  usable_ratio: 0.85,
  recent_ratio: 0.25,
  semantic_ratio: 0.35,
  durable_memory_ratio: 0.15,
  document_ratio: 0.2,
  memory_expanded_ratio: 0.05,
};

const source = readFileSync(new URL('../src/lib/context-budget.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const {
  sliceTokenBudget,
  llmFreeTokens,
  prefetchCapTokens,
  answerReserveTokens,
  headroomTokens,
} = await import('data:text/javascript,' + encodeURIComponent(compiled));

test('slice token label matches context size × usable × slice ratio', () => {
  const contextWindow = 8192;
  const tokens = sliceTokenBudget(contextWindow, TEST_RATIOS, 'recent_ratio');
  const expected = Math.round(contextWindow * TEST_RATIOS.usable_ratio * TEST_RATIOS.recent_ratio);
  assert.equal(tokens, expected);
});

test('llm free tokens equal headroom plus answer reserve', () => {
  const contextWindow = 60000;
  const free = llmFreeTokens(contextWindow, TEST_RATIOS);
  const expected =
    headroomTokens(contextWindow, TEST_RATIOS)
    + answerReserveTokens(contextWindow, TEST_RATIOS);
  assert.equal(free, expected);
  assert.equal(free + prefetchCapTokens(contextWindow, TEST_RATIOS), contextWindow);
  assert.ok(free > 0);
  assert.ok(free < contextWindow);
});
