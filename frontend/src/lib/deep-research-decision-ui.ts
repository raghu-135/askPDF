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

/** Mirrors `_OBJECTIVE_WEB_EVIDENCE_PATTERN` in deep_research_nodes.py */
const OBJECTIVE_WEB_EVIDENCE_PATTERN = /\b(web|benchmark|industry\s+comparison|compare\s+.*\s+to\s+typical|live\s+web|external\s+sources?)\b/i;

export function objectiveRequestsLiveWeb(objective: string): boolean {
  return OBJECTIVE_WEB_EVIDENCE_PATTERN.test(String(objective || ''));
}

const WAIVE_WEB_GUIDANCE_PATTERN = /\b(skip|omit|ignore|waive|without|no\s+need\s+for)\b[\s\S]{0,40}\b(web|benchmarks?|industry\s+comparison)/i;

export function guidanceWaivesWebBenchmarks(...texts: string[]): boolean {
  return texts.some((text) => WAIVE_WEB_GUIDANCE_PATTERN.test(String(text || '')));
}

export function effectiveObjectiveRequestsLiveWeb(
  objective: string,
  guidanceTexts: string[] = [],
): boolean {
  if (guidanceWaivesWebBenchmarks(...guidanceTexts)) return false;
  const combined = [objective, ...guidanceTexts].filter(Boolean).join('\n');
  return objectiveRequestsLiveWeb(combined);
}

const LIMITATION_TOPIC_PATTERNS: Array<{ topic: string; pattern: RegExp }> = [
  { topic: 'index_readiness', pattern: /document\s+index\s+(?:is\s+)?not\s+(\w+\s+)*ready|index(?:ing)?\s+(?:is\s+)?not\s+ready/i },
  { topic: 'staff_absence', pattern: /\bno\s+staff\b|staff\s+information\s+not|without\s+staff|does\s+not\s+provide\s+(?:any\s+)?staff/i },
  { topic: 'web_off', pattern: /web\s+search\s+is\s+off|web\s+benchmarks?/i },
];

function limitationTopics(text: string): Set<string> {
  const topics = new Set<string>();
  LIMITATION_TOPIC_PATTERNS.forEach(({ topic, pattern }) => {
    if (pattern.test(text)) topics.add(topic);
  });
  return topics;
}

export type ReviewLimitationItem = { kind: 'gap' | 'finding'; text: string };

function normalizeLimitationKey(text: string): string {
  return text.toLowerCase().replace(/[^\w\s]+/g, ' ').replace(/\s+/g, ' ').trim();
}

function isRedundantFinding(text: string, gapKeys: Set<string>, gapTopics: Set<string>): boolean {
  const issueKey = normalizeLimitationKey(text);
  const issueTopics = limitationTopics(text);
  if (issueTopics.size && [...issueTopics].every((topic) => gapTopics.has(topic))) {
    return true;
  }
  if (!issueKey) return true;
  for (const gapKey of gapKeys) {
    if (!gapKey) continue;
    if (issueKey === gapKey) return true;
    if (issueKey.length >= 8 && gapKey.includes(issueKey)) return true;
    if (gapKey.length >= 8 && issueKey.includes(gapKey)) return true;
  }
  return false;
}

export function normalizeReviewGapText(
  gap: unknown,
  todoTitles?: Record<string, string>,
): string {
  if (gap && typeof gap === 'object') {
    const record = gap as { message?: string; summary?: string; text?: string; code?: string };
    const message = String(record.message || record.summary || record.text || '').trim();
    if (message) return enrichReviewGapLine(message, todoTitles);
    const code = String(record.code || '').trim();
    if (code) return enrichReviewGapLine(code.replaceAll('_', ' '), todoTitles);
    return '';
  }
  return enrichReviewGapLine(String(gap || ''), todoTitles);
}

const REVIEW_FINDING_CODE_LABELS: Record<string, string> = {
  INSUFFICIENT_EVIDENCE: 'The available evidence does not support a confident answer for this objective.',
  MISSING_CONTENT: 'Required content was missing from the sources that were reviewed.',
  MISSING_EVIDENCE: 'Expected evidence was not found in the retrieved sources.',
  INCOMPLETE_EVIDENCE: 'The retrieved evidence is incomplete for this objective.',
  UNSUPPORTED_CLAIM: 'The report includes claims that are not adequately supported by cited evidence.',
  evidence_critic_issues: 'Evidence review identified issues that need attention.',
};

export function humanizeReviewFindingCode(code: string): string {
  const trimmed = String(code || '').trim();
  if (!trimmed) return trimmed;
  const underscored = trimmed.toUpperCase().replace(/\s+/g, '_');
  if (REVIEW_FINDING_CODE_LABELS[underscored]) return REVIEW_FINDING_CODE_LABELS[underscored];
  if (REVIEW_FINDING_CODE_LABELS[trimmed]) return REVIEW_FINDING_CODE_LABELS[trimmed];
  const words = trimmed.replace(/_/g, ' ').toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function humanizeReviewFindingText(text: string): string {
  const trimmed = String(text || '').trim();
  if (!trimmed) return trimmed;
  if (/^[A-Z][A-Z0-9_ ]+$/.test(trimmed)) return humanizeReviewFindingCode(trimmed);
  return trimmed;
}

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

export function reviewLimitationItems(
  pending: AgentRunPendingInterrupt | null | undefined,
  todoTitles?: Record<string, string>,
): ReviewLimitationItem[] {
  if (!pending) return [];
  const items: ReviewLimitationItem[] = [];
  const gapKeys = new Set<string>();
  const gapTopics = new Set<string>();
  const gaps = Array.isArray(pending.gaps) ? pending.gaps : [];
  gaps.forEach((gap) => {
    const enriched = normalizeReviewGapText(gap, todoTitles);
    if (!enriched) return;
    items.push({ kind: 'gap', text: enriched });
    gapKeys.add(normalizeLimitationKey(enriched));
    gapKeys.add(normalizeLimitationKey(gap));
    limitationTopics(enriched).forEach((topic) => gapTopics.add(topic));
    limitationTopics(typeof gap === 'string' ? gap : enriched).forEach((topic) => gapTopics.add(topic));
  });
  const warnings = Array.isArray(pending.warnings) ? pending.warnings : [];
  warnings.forEach((entry) => {
    if (!entry || typeof entry !== 'object') return;
    const code = String((entry as { code?: string }).code || 'warning');
    const details = (entry as { details?: Record<string, unknown> }).details;
    const issues = details && Array.isArray(details.issues) ? details.issues : [];
    if (issues.length) {
      issues.forEach((issue) => {
        let message = '';
        if (typeof issue === 'string') {
          message = humanizeReviewFindingText(issue);
        } else if (issue && typeof issue === 'object') {
          const raw = String(
            (issue as { message?: string }).message
            || (issue as { summary?: string }).summary
            || (issue as { code?: string }).code
            || code,
          );
          message = humanizeReviewFindingText(raw);
        }
        if (!message || isNoiseCriticIssue(message) || isRedundantFinding(message, gapKeys, gapTopics)) return;
        items.push({ kind: 'finding', text: message });
      });
    } else if (code !== 'evidence_critic_issues') {
      const fallback = humanizeReviewFindingCode(code);
      if (!isRedundantFinding(fallback, gapKeys, gapTopics)) {
        items.push({ kind: 'finding', text: fallback });
      }
    }
  });
  return items;
}

export function reviewSummaryLines(
  pending: AgentRunPendingInterrupt | null | undefined,
  todoTitles?: Record<string, string>,
): string[] {
  return reviewLimitationItems(pending, todoTitles).map((item) => item.text);
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
  const items = reviewLimitationItems(pending, todoTitles);
  const gapCount = items.filter((item) => item.kind === 'gap').length;
  const reviewFindingCount = items.filter((item) => item.kind === 'finding').length;
  const warnings = Array.isArray(pending?.warnings) ? pending!.warnings : [];
  return {
    warningCount: warnings.length,
    gapCount,
    reviewFindingCount,
    limitationCount: items.length,
    total: items.length,
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
    return `Research budget for ${names} is at the tranche limit.${breakdown} You can finish with this result or continue with guidance (starts a new budget tranche).`;
  }
  if (limitationCount > 0) {
    return `Open research issues remain.${breakdown} Finish with this result or continue with guidance for another tranche.`;
  }
  return fallbackBody || 'Review the provisional answer before you continue or finish.';
}

export function resultReviewContinueLabel(limitationCount: number): string {
  if (limitationCount > 0) {
    return 'Continue with guidance (another tranche)';
  }
  return 'Continue with guidance';
}

export const RESEARCH_BUDGET_TOOLTIP =
  'Counts per tranche: planner, each subagent worker, synthesizer, and critic each use one model call when metering is active.';
