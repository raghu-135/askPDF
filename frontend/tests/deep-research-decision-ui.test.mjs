import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildDecisionLeadLine,
  budgetUsageFromInterrupt,
  countReviewIssues,
  formatBudgetMeterLine,
} from '../src/lib/deep-research-decision-ui.ts';

test('budgetUsageFromInterrupt reads tranche usage from pending usage payload', () => {
  const snapshot = budgetUsageFromInterrupt({
    usage: {
      tranche_index: 2,
      tranche_limits: { model_calls: 3, model_tokens: 1000, tool_calls: 10, elapsed_active_ms: 60000 },
      tranche_usage: { model_calls: 3, model_tokens: 200, tool_calls: 1, elapsed_active_ms: 1000 },
      exhausted_dimensions: ['model_calls'],
    },
  }, null);
  assert.equal(snapshot?.trancheIndex, 2);
  assert.deepEqual(snapshot?.exhaustedDimensions, ['model_calls']);
  assert.match(formatBudgetMeterLine(snapshot), /Model calls 3\/3/);
});

test('buildDecisionLeadLine mentions budget exhaustion for result review', () => {
  const line = buildDecisionLeadLine({
    isBudgetReview: false,
    snapshot: {
      trancheIndex: 1,
      trancheUsage: { model_calls: 3 },
      trancheLimits: { model_calls: 3 },
      exhaustedDimensions: ['model_calls'],
    },
    issueCount: 2,
    gapCount: 1,
  });
  assert.match(line, /Research budget for model calls/i);
  assert.match(line, /more budget/i);
});

test('countReviewIssues totals warnings and gaps', () => {
  const counts = countReviewIssues({
    warnings: [{ code: 'a' }, { code: 'b' }],
    gaps: ['gap-one'],
  });
  assert.equal(counts.warningCount, 2);
  assert.equal(counts.gapCount, 1);
  assert.equal(counts.total, 3);
});
