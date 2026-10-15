export type ContextBudgetRatios = {
  usable_ratio: number;
  recent_ratio: number;
  semantic_ratio: number;
  durable_memory_ratio: number;
  document_ratio: number;
  memory_expanded_ratio: number;
};

export const FALLBACK_CONTEXT_BUDGET: ContextBudgetRatios = {
  usable_ratio: 0.8,
  recent_ratio: 0.22,
  semantic_ratio: 0.12,
  durable_memory_ratio: 0.06,
  document_ratio: 0.28,
  memory_expanded_ratio: 0.12,
};

export const CONTEXT_BUDGET_STORAGE_KEY = 'last_context_budget';

export function packingBudgetPayload(budget: ContextBudgetRatios) {
  return {
    usable_ratio: budget.usable_ratio,
    recent_ratio: budget.recent_ratio,
    semantic_ratio: budget.semantic_ratio,
    durable_memory_ratio: budget.durable_memory_ratio,
    document_ratio: budget.document_ratio,
    memory_expanded_ratio: budget.memory_expanded_ratio,
  };
}

export const PACKING_SLICES: Array<{
  key: keyof Pick<ContextBudgetRatios, 'recent_ratio' | 'semantic_ratio' | 'durable_memory_ratio' | 'document_ratio'>;
  label: string;
}> = [
  { key: 'recent_ratio', label: 'Recent turns' },
  { key: 'semantic_ratio', label: 'Semantic memory' },
  { key: 'durable_memory_ratio', label: 'Durable memory' },
  { key: 'document_ratio', label: 'Documents' },
];

export function sliceTokenBudget(contextWindow: number, ratios: ContextBudgetRatios, sliceKey: keyof ContextBudgetRatios): number {
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
  if (code === 'context_length_exceeded') {
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
    const parsed = JSON.parse(raw) as ContextBudgetRatios;
    if (!parsed || typeof parsed !== 'object') return null;
    return parsed;
  } catch {
    return null;
  }
}

export function writeStoredContextBudget(ratios: ContextBudgetRatios): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(CONTEXT_BUDGET_STORAGE_KEY, JSON.stringify(ratios));
}
