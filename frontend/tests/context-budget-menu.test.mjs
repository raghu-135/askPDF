import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/context-budget.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const {
  sliceTokenBudget,
  FALLBACK_CONTEXT_BUDGET,
  llmFreeTokens,
  prefetchCapTokens,
  answerReserveTokens,
  headroomTokens,
} = await import('data:text/javascript,' + encodeURIComponent(compiled));

test('slice token label matches context size × usable × slice ratio', () => {
  const contextWindow = 8192;
  const tokens = sliceTokenBudget(contextWindow, FALLBACK_CONTEXT_BUDGET, 'recent_ratio');
  const expected = Math.round(contextWindow * FALLBACK_CONTEXT_BUDGET.usable_ratio * FALLBACK_CONTEXT_BUDGET.recent_ratio);
  assert.equal(tokens, expected);
});

test('llm free tokens equal headroom plus answer reserve', () => {
  const contextWindow = 60000;
  const free = llmFreeTokens(contextWindow, FALLBACK_CONTEXT_BUDGET);
  const expected =
    headroomTokens(contextWindow, FALLBACK_CONTEXT_BUDGET)
    + answerReserveTokens(contextWindow, FALLBACK_CONTEXT_BUDGET);
  assert.equal(free, expected);
  assert.equal(free + prefetchCapTokens(contextWindow, FALLBACK_CONTEXT_BUDGET), contextWindow);
  assert.ok(free > 0);
  assert.ok(free < contextWindow);
});
