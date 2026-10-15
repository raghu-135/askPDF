import type { ContextBudgetRatios } from './context-budget';

export const CONTEXT_PACKING_MENU_HELP =
  'Controls how much history, memory, and document text AskPDF tries to load before each model call. This is separate from the provider’s real context limit—stay at or below what your model supports.';

export const CONTEXT_SIZE_HELP =
  'Your token target for one run (prompt + answer). If you see context-length errors, lower this number or reduce the retrieval sliders.';

export const CONTEXT_WINDOW_SPLIT_HELP =
  'The bar shows how that target is divided. Orange is the max retrieval budget (your sliders). Blue is room reserved for system instructions and the reply. Green is extra margin left empty on purpose.';

export const CONTEXT_PACKING_WINDOW_HELP =
  'Fraction of ctx size used when calculating the slices. The remainder stays as safety margin (green in the bar) so packing does not assume the full window.';

export const CONTEXT_COMPACT_BAR_HELP =
  'Orange: max history, memory, and PDF text AskPDF may load. Green: everything else—system prompt, tools, reply, and safety margin. Hover the bar or tap the help icon for details.';

export const SLICE_HELP: Record<
  keyof Pick<ContextBudgetRatios, 'recent_ratio' | 'semantic_ratio' | 'durable_memory_ratio' | 'document_ratio'>,
  string
> = {
  recent_ratio:
    'Cap on recent user and assistant turns included in the prompt. Raise for long threads; lower to leave more room for documents.',
  semantic_ratio:
    'Cap on semantically retrieved past Q/A from this thread (topic recall, not strict chronology).',
  durable_memory_ratio:
    'Cap on saved durable memories (preferences, facts) from allowed scopes (thread, project, user).',
  document_ratio:
    'Cap on retrieved PDF and document text. Raise when answers need more source material; lower if prompts overflow.',
};
