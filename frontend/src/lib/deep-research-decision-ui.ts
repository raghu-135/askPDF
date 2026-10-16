import type { AgentRunPendingInterrupt, AgentTaskSummary } from './api';

export const RESEARCH_BUDGET_DIMENSIONS = [
  'model_calls',
  'model_tokens',
  'tool_calls',
  'elapsed_active_ms',
] as const;

export type ResearchBudgetDimension = (typeof RESEARCH_BUDGET_DIMENSIONS)[number];

const DIMENSION_LABELS: Record<ResearchBudgetDimension, string> = {
  model_calls: 'Model calls',
  model_tokens: 'Model tokens',
  tool_calls: 'Tool calls',
  elapsed_active_ms: 'Active runtime',
};

export function researchBudgetDimensionLabel(dimension: string): string {
  return DIMENSION_LABELS[dimension as ResearchBudgetDimension] || dimension.replaceAll('_', ' ');
}

export type BudgetUsageSnapshot = {
  trancheIndex: number;
  trancheUsage: Record<string, number>;
  trancheLimits: Record<string, number>;
  exhaustedDimensions: string[];
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asNumberMap(value: unknown): Record<string, number> {
  const source = asRecord(value);
  const result: Record<string, number> = {};
  Object.entries(source).forEach(([key, raw]) => {
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) result[key] = parsed;
  });
  return result;
}

export function budgetUsageFromTask(task: AgentTaskSummary | null | undefined): BudgetUsageSnapshot | null {
  if (!task?.budgets) return null;
  const budgets = asRecord(task.budgets);
  const trancheLimits = asNumberMap(budgets.tranche_limits);
  const trancheUsage = asNumberMap(budgets.tranche_usage);
  if (!Object.keys(trancheLimits).length) return null;
  const boundary = asRecord(budgets.boundary);
  const exhausted = Array.isArray(boundary.dimensions)
    ? boundary.dimensions.map(String)
    : RESEARCH_BUDGET_DIMENSIONS.filter((key) => (trancheUsage[key] || 0) >= (trancheLimits[key] || 1));
  return {
    trancheIndex: Math.max(1, Number(budgets.tranche_index) || 1),
    trancheUsage,
    trancheLimits,
    exhaustedDimensions: exhausted,
  };
}

export function budgetUsageFromInterrupt(
  pending: AgentRunPendingInterrupt | null | undefined,
  task: AgentTaskSummary | null | undefined,
): BudgetUsageSnapshot | null {
  const usage = asRecord(pending?.usage);
  if (Object.keys(usage).length) {
    const trancheLimits = asNumberMap(usage.tranche_limits);
    const trancheUsage = asNumberMap(usage.tranche_usage);
    const exhausted = Array.isArray(usage.exhausted_dimensions)
      ? usage.exhausted_dimensions.map(String)
      : [];
    if (Object.keys(trancheLimits).length) {
      return {
        trancheIndex: Math.max(1, Number(usage.tranche_index) || 1),
        trancheUsage,
        trancheLimits,
        exhaustedDimensions: exhausted.length
          ? exhausted
          : RESEARCH_BUDGET_DIMENSIONS.filter((key) => (trancheUsage[key] || 0) >= (trancheLimits[key] || 1)),
      };
    }
  }
  return budgetUsageFromTask(task);
}

export function countReviewIssues(
  pending: AgentRunPendingInterrupt | null | undefined,
): { warningCount: number; gapCount: number; total: number } {
  const warnings = Array.isArray(pending?.warnings) ? pending!.warnings : [];
  const gaps = Array.isArray(pending?.gaps) ? pending!.gaps.map(String) : [];
  const warningCount = warnings.length;
  const gapCount = gaps.filter((gap) => gap.trim()).length;
  return { warningCount, gapCount, total: warningCount + gapCount };
}

export function formatBudgetMeterLine(snapshot: BudgetUsageSnapshot): string {
  const parts = RESEARCH_BUDGET_DIMENSIONS.map((key) => {
    const used = snapshot.trancheUsage[key] || 0;
    const limit = snapshot.trancheLimits[key];
    if (limit == null) return null;
    return `${researchBudgetDimensionLabel(key)} ${used}/${limit}`;
  }).filter(Boolean);
  return parts.join(' · ');
}

export function buildDecisionLeadLine(options: {
  isBudgetReview: boolean;
  snapshot: BudgetUsageSnapshot | null;
  issueCount: number;
  gapCount: number;
  fallbackBody?: string | null;
}): string {
  const { isBudgetReview, snapshot, issueCount, gapCount, fallbackBody } = options;
  const exhausted = snapshot?.exhaustedDimensions || [];
  if (isBudgetReview && exhausted.length) {
    const names = exhausted.map(researchBudgetDimensionLabel).join(', ');
    return `This tranche used its research budget (${names}). Review the provisional answer before spending more budget.`;
  }
  if (!isBudgetReview && exhausted.length) {
    const names = exhausted.map(researchBudgetDimensionLabel).join(', ');
    const issuePart = issueCount > 0 || gapCount > 0
      ? ` ${issueCount} claim${issueCount === 1 ? '' : 's'} flagged · ${gapCount} gap${gapCount === 1 ? '' : 's'}.`
      : '';
    return `Research budget for ${names} is at the tranche limit.${issuePart} You can finish with this result or continue with guidance (uses more budget).`;
  }
  if (issueCount > 0 || gapCount > 0) {
    return `${issueCount} unverified claim${issueCount === 1 ? '' : 's'} and ${gapCount} open gap${gapCount === 1 ? '' : 's'}. Finish with this result or continue with guidance (uses more budget).`;
  }
  return fallbackBody || 'Review the provisional answer before you continue or finish.';
}

export const RESEARCH_BUDGET_TOOLTIP =
  'Counts per tranche: planner, each subagent worker, synthesizer, and critic each use one model call when metering is active.';
