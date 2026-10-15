import type { ContextBudgetDefaults } from './api';

export const CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded';

export const PACKING_RATIO_KEYS = [
  'usable_ratio',
  'recent_ratio',
  'semantic_ratio',
  'durable_memory_ratio',
  'document_ratio',
  'memory_expanded_ratio',
] as const;

export type ContextBudgetRatios = Pick<ContextBudgetDefaults, typeof PACKING_RATIO_KEYS[number]>;

export type PrefetchSliceKey = keyof Pick<
  ContextBudgetRatios,
  'recent_ratio' | 'semantic_ratio' | 'durable_memory_ratio' | 'document_ratio'
>;

export const CONTEXT_BUDGET_STORAGE_KEY = 'last_context_budget';

function isOpenUnitRatio(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= 1;
}

export function contextBudgetFromDefaults(defaults: ContextBudgetDefaults): ContextBudgetRatios {
  return {
    usable_ratio: defaults.usable_ratio,
    recent_ratio: defaults.recent_ratio,
    semantic_ratio: defaults.semantic_ratio,
    durable_memory_ratio: defaults.durable_memory_ratio,
    document_ratio: defaults.document_ratio,
    memory_expanded_ratio: defaults.memory_expanded_ratio,
  };
}

export function normalizeStoredContextBudget(raw: unknown): ContextBudgetRatios | null {
  if (!raw || typeof raw !== 'object') return null;
  const candidate = raw as Record<string, unknown>;
  for (const key of PACKING_RATIO_KEYS) {
    if (!isOpenUnitRatio(candidate[key])) return null;
  }
  const ratios = contextBudgetFromDefaults(candidate as ContextBudgetDefaults);
  if (prefetchSliceSum(ratios) > 1) return null;
  return ratios;
}

export function packingBudgetPayload(budget: ContextBudgetRatios): ContextBudgetRatios {
  const payload: Partial<ContextBudgetRatios> = {};
  for (const key of PACKING_RATIO_KEYS) {
    payload[key] = budget[key];
  }
  return payload as ContextBudgetRatios;
}

export const PACKING_SLICES: Array<{ key: PrefetchSliceKey; label: string }> = [
  { key: 'recent_ratio', label: 'Recent turns' },
  { key: 'semantic_ratio', label: 'Semantic memory' },
  { key: 'durable_memory_ratio', label: 'Durable memory' },
  { key: 'document_ratio', label: 'Documents' },
];

export function sliceTokenBudget(
  contextWindow: number,
  ratios: ContextBudgetRatios,
  sliceKey: PrefetchSliceKey,
): number {
  const sliceRatio = ratios[sliceKey];
  return Math.max(0, Math.round(contextWindow * ratios.usable_ratio * sliceRatio));
}

export function prefetchSliceSum(ratios: ContextBudgetRatios): number {
  return ratios.recent_ratio + ratios.semantic_ratio + ratios.durable_memory_ratio + ratios.document_ratio;
}

export function answerReserveTokens(contextWindow: number, ratios: ContextBudgetRatios): number {
  const assigned = prefetchSliceSum(ratios);
  const reserveShare = Math.max(0, 1 - assigned);
  return Math.max(0, Math.round(contextWindow * ratios.usable_ratio * reserveShare));
}

export function usablePackingTokens(contextWindow: number, ratios: ContextBudgetRatios): number {
  return Math.max(0, Math.round(contextWindow * ratios.usable_ratio));
}

export function prefetchCapTokens(contextWindow: number, ratios: ContextBudgetRatios): number {
  return Math.max(0, Math.round(contextWindow * ratios.usable_ratio * prefetchSliceSum(ratios)));
}

export function headroomTokens(contextWindow: number, ratios: ContextBudgetRatios): number {
  return Math.max(0, contextWindow - usablePackingTokens(contextWindow, ratios));
}

/** Tokens not capped for prefetch slices: answer/system reserve plus unused window headroom. */
export function llmFreeTokens(contextWindow: number, ratios: ContextBudgetRatios): number {
  return headroomTokens(contextWindow, ratios) + answerReserveTokens(contextWindow, ratios);
}

export type ContextWindowBreakdown = {
  prefetchCap: number;
  answerReserve: number;
  headroom: number;
  llmFree: number;
};

export function contextWindowBreakdown(
  contextWindow: number,
  ratios: ContextBudgetRatios,
): ContextWindowBreakdown {
  const answerReserve = answerReserveTokens(contextWindow, ratios);
  const headroom = headroomTokens(contextWindow, ratios);
  const prefetchCap = prefetchCapTokens(contextWindow, ratios);
  return {
    prefetchCap,
    answerReserve,
    headroom,
    llmFree: headroom + answerReserve,
  };
}

export function formatContextLengthUserMessage(error: unknown): { title: string; detail?: string } {
  const payload = error && typeof error === 'object' ? error as { code?: string; raw_message?: string; message?: string } : {};
  const code = String(payload.code || '');
  const detail = String(payload.raw_message || payload.message || error || '').trim();
  if (code === CONTEXT_LENGTH_EXCEEDED) {
    return {
      title: 'The model rejected this prompt because it exceeded the model context window. Lower Ctx size or reduce the document and history sliders, then try again.',
      detail: detail || undefined,
    };
  }
  return { title: detail || 'Chat workflow failed.' };
}

export function readStoredContextBudget(): ContextBudgetRatios | null {
  if (typeof window === 'undefined') return null;
  const raw = window.localStorage.getItem(CONTEXT_BUDGET_STORAGE_KEY);
  if (!raw) return null;
  try {
    return normalizeStoredContextBudget(JSON.parse(raw));
  } catch {
    return null;
  }
}

export function writeStoredContextBudget(ratios: ContextBudgetRatios): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(CONTEXT_BUDGET_STORAGE_KEY, JSON.stringify(ratios));
}
