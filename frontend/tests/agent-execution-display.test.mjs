import assert from 'node:assert/strict';
import test from 'node:test';

import { compactExecutionText } from '../src/components/agent-graph/agent-execution-display.ts';

test('compacts markdown-like execution summaries without changing source data', () => {
  const source = '**Selected the execute route:**\n\nRun document retrieval.';
  assert.equal(compactExecutionText(source), 'Selected the execute route: Run document retrieval.');
  assert.equal(source, '**Selected the execute route:**\n\nRun document retrieval.');
});

test('bounds display text and handles structured values', () => {
  assert.equal(compactExecutionText('abcdefghij', 6), 'abcde…');
  assert.equal(compactExecutionText({ route: 'execute' }), '{"route":"execute"}');
});
