import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildDecisionLeadLine,
  budgetUsageFromInterrupt,
  countReviewIssues,
  formatBudgetDimensionChip,
  formatBudgetMeterLine,
  objectiveRequestsLiveWeb,
  reviewLimitationItems,
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
    gapCount: 5,
    reviewFindingCount: 3,
    limitationCount: 8,
  });
  assert.match(line, /Research budget for model calls/i);
  assert.match(line, /8 limitations/i);
  assert.match(line, /5 gaps/i);
});

test('formatBudgetDimensionChip formats active runtime as duration', () => {
  assert.match(formatBudgetDimensionChip('elapsed_active_ms', 24885, 7200000), /25s/);
});

test('countReviewIssues totals structured limitation lines', () => {
  const counts = countReviewIssues({
    warnings: [{ code: 'evidence_critic_issues', details: { issues: ['issue-a', 'issue-b'] } }],
    gaps: ['gap-one'],
  });
  assert.equal(counts.warningCount, 1);
  assert.equal(counts.gapCount, 1);
  assert.equal(counts.reviewFindingCount, 2);
  assert.equal(counts.limitationCount, 3);
  assert.equal(counts.total, 3);
});

test('reviewLimitationItems dedupes critic issues that repeat gaps', () => {
  const gap = 'Web benchmarks or live web evidence were requested in the objective, but web search is off for this task.';
  const items = reviewLimitationItems({
    gaps: [gap],
    warnings: [{ code: 'evidence_critic_issues', details: { issues: [gap, 'Unique critic finding'] } }],
  });
  assert.equal(items.length, 2);
  assert.equal(items[0].kind, 'gap');
  assert.equal(items[1].kind, 'finding');
  assert.equal(items[1].text, 'Unique critic finding');
});

test('objectiveRequestsLiveWeb detects benchmark objectives', () => {
  assert.equal(objectiveRequestsLiveWeb('Compare to typical US bakery benchmarks using web sources'), true);
  assert.equal(objectiveRequestsLiveWeb('Summarize the PDF only'), false);
});
