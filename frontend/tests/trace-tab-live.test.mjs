import assert from 'node:assert/strict';
import test from 'node:test';

import { isAgentRunActivelyStreaming, isTraceTabLive } from '../src/lib/trace-tab-live.ts';

test('live chip only tracks actively running agent runs', () => {
  assert.equal(isAgentRunActivelyStreaming('running'), true);
  assert.equal(isAgentRunActivelyStreaming('awaiting_human'), false);
  assert.equal(isAgentRunActivelyStreaming('completed'), false);
});

test('trace tab live ignores stale running flag after completion', () => {
  assert.equal(isTraceTabLive({
    running: true,
    status: 'completed',
    runDetails: { status: 'completed' },
  }), false);
  assert.equal(isTraceTabLive({
    running: true,
    status: 'running',
    runDetails: { status: 'running' },
  }), true);
});
