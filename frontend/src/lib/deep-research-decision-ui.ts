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

const DIMENSION_TOOLTIPS: Record<ResearchBudgetDimension, string> = {
  model_calls: 'Planner, subagent workers, synthesizer, and critic model calls in this tranche.',
  model_tokens: 'Total model input and output tokens charged in this tranche.',
  tool_calls: 'Document, web, and memory tool invocations in this tranche.',
  elapsed_active_ms: 'Active runtime while research workers were executing in this tranche.',
};

export function researchBudgetDimensionTooltip(dimension: string): string {
  return DIMENSION_TOOLTIPS[dimension as ResearchBudgetDimension] || RESEARCH_BUDGET_TOOLTIP;
}

export function formatResearchDurationMs(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return remMinutes ? `${hours}h ${remMinutes}m` : `${hours}h`;
}

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

const TODO_GAP_PATTERN = /^todo-\d+$/i;

export function enrichReviewGapLine(
  gap: string,
  todoTitles?: Record<string, string>,
): string {
  const trimmed = gap.trim();
  if (!trimmed) return trimmed;
  const todoMatch = trimmed.match(/^(todo-\d+)(?::\s*(.*))?$/i);
  if (todoMatch) {
    const id = todoMatch[1];
    const rest = (todoMatch[2] || '').trim();
    const title = todoTitles?.[id];
    if (title) {
      return rest && rest !== 'failed' && rest !== 'pending'
        ? `${title} (${id}): ${rest}`
        : `${title} (${id})`;
    }
  }
  if (TODO_GAP_PATTERN.test(trimmed) && todoTitles?.[trimmed]) {
    return `${todoTitles[trimmed]} (${trimmed})`;
  }
  return trimmed;
}

export function reviewSummaryLines(
  pending: AgentRunPendingInterrupt | null | undefined,
  todoTitles?: Record<string, string>,
): string[] {
  if (!pending) return [];
  const lines: string[] = [];
  const gaps = Array.isArray(pending.gaps) ? pending.gaps.map(String) : [];
  gaps.forEach((gap) => {
    const enriched = enrichReviewGapLine(gap, todoTitles);
    if (enriched) lines.push(enriched);
  });
  const warnings = Array.isArray(pending.warnings) ? pending.warnings : [];
  warnings.forEach((entry) => {
    if (!entry || typeof entry !== 'object') return;
    const code = String((entry as { code?: string }).code || 'warning');
    const details = (entry as { details?: Record<string, unknown> }).details;
    const issues = details && Array.isArray(details.issues) ? details.issues : [];
    if (issues.length) {
      issues.forEach((issue) => {
        if (typeof issue === 'string') {
          if (!isNoiseCriticIssue(issue)) lines.push(issue);
        } else if (issue && typeof issue === 'object') {
          const message = String(
            (issue as { message?: string }).message
            || (issue as { summary?: string }).summary
            || code,
          );
          if (!isNoiseCriticIssue(message)) lines.push(message);
        }
      });
    } else {
      lines.push(code.replaceAll('_', ' '));
    }
  });
  return lines;
}

export function isNoiseCriticIssue(text: string): boolean {
  const lowered = text.toLowerCase();
  return (
    lowered.includes('not smuggled instruction')
    || lowered.includes('are not smuggled instructions')
    || lowered.includes('not prompt injection')
  );
}

export function countReviewIssues(
  pending: AgentRunPendingInterrupt | null | undefined,
  todoTitles?: Record<string, string>,
): { warningCount: number; gapCount: number; reviewFindingCount: number; limitationCount: number; total: number } {
  const lines = reviewSummaryLines(pending, todoTitles);
  const gaps = Array.isArray(pending?.gaps) ? pending!.gaps.map(String).filter((gap) => gap.trim()) : [];
  const gapLineCount = gaps.length;
  const reviewFindingCount = Math.max(0, lines.length - gapLineCount);
  const warnings = Array.isArray(pending?.warnings) ? pending!.warnings : [];
  return {
    warningCount: warnings.length,
    gapCount: gaps.length,
    reviewFindingCount,
    limitationCount: lines.length,
    total: lines.length,
  };
}

export function formatBudgetDimensionChip(
  dimension: string,
  used: number,
  limit: number,
): string {
  if (dimension === 'elapsed_active_ms') {
    return `${researchBudgetDimensionLabel(dimension)} ${formatResearchDurationMs(used)}/${formatResearchDurationMs(limit)}`;
  }
  return `${researchBudgetDimensionLabel(dimension)} ${used}/${limit}`;
}

export function formatBudgetMeterLine(snapshot: BudgetUsageSnapshot): string {
  const parts = RESEARCH_BUDGET_DIMENSIONS.map((key) => {
    const used = snapshot.trancheUsage[key] || 0;
    const limit = snapshot.trancheLimits[key];
    if (limit == null) return null;
    return formatBudgetDimensionChip(key, used, limit);
  }).filter(Boolean);
  return parts.join(' · ');
}

function limitationBreakdownPhrase(gapCount: number, reviewFindingCount: number, limitationCount: number): string {
  if (limitationCount <= 0) return '';
  if (gapCount > 0 && reviewFindingCount > 0) {
    return ` ${limitationCount} limitation${limitationCount === 1 ? '' : 's'} (${gapCount} gap${gapCount === 1 ? '' : 's'} · ${reviewFindingCount} review finding${reviewFindingCount === 1 ? '' : 's'}).`;
  }
  if (limitationCount > 0) {
    return ` ${limitationCount} limitation${limitationCount === 1 ? '' : 's'} listed below.`;
  }
  return '';
}

export function buildDecisionLeadLine(options: {
  isBudgetReview: boolean;
  snapshot: BudgetUsageSnapshot | null;
  gapCount: number;
  reviewFindingCount: number;
  limitationCount: number;
  fallbackBody?: string | null;
}): string {
  const {
    isBudgetReview,
    snapshot,
    gapCount,
    reviewFindingCount,
    limitationCount,
    fallbackBody,
  } = options;
  const exhausted = snapshot?.exhaustedDimensions || [];
  const breakdown = limitationBreakdownPhrase(gapCount, reviewFindingCount, limitationCount);
  if (isBudgetReview && exhausted.length) {
    const names = exhausted.map(researchBudgetDimensionLabel).join(', ');
    return `This tranche used its research budget (${names}).${breakdown} Review the provisional answer before spending more budget.`;
  }
  if (!isBudgetReview && exhausted.length) {
    const names = exhausted.map(researchBudgetDimensionLabel).join(', ');
    return `Research budget for ${names} is at the tranche limit.${breakdown} You can finish with this result or continue with guidance (uses more budget).`;
  }
  if (limitationCount > 0) {
    return `Open research issues remain.${breakdown} Finish with this result or continue with guidance (uses more budget).`;
  }
  return fallbackBody || 'Review the provisional answer before you continue or finish.';
}

export const RESEARCH_BUDGET_TOOLTIP =
  'Counts per tranche: planner, each subagent worker, synthesizer, and critic each use one model call when metering is active.';
