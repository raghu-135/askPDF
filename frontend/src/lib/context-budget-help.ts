import type { ContextBudgetRatios } from './context-budget';

export const CONTEXT_PACKING_MENU_HELP =
  'Controls how much history, memory, and document text AskPDF tries to load before each model call. This is separate from the provider’s real context limit—stay at or below what your model supports.';

export const CONTEXT_SIZE_HELP =
  'Your token target for one run (prompt + answer). If you see context-length errors, lower this number or reduce the retrieval sliders.';

export const CONTEXT_WINDOW_SPLIT_HELP =
  'Hover each color on the bar: orange = max retrieval (your sliders), blue = system + answer, green = safety margin.';

export const CONTEXT_PACKING_WINDOW_HELP =
  'Fraction of ctx size used when calculating the slices. The remainder stays as safety margin (green in the bar) so packing does not assume the full window.';

export const BAR_SEGMENT_RETRIEVAL_TITLE = 'Orange — max retrieval';
export const BAR_SEGMENT_RETRIEVAL_BODY =
  'Upper limit on history, memory, and PDF text AskPDF may load before the model runs. Set by the four sliders above.';

export const BAR_SEGMENT_REPLY_TITLE = 'Blue — system + answer';
export const BAR_SEGMENT_REPLY_BODY =
  'Room inside the packing window reserved for system instructions, tools, and the model’s reply when your slice sliders do not use 100%.';

export const BAR_SEGMENT_SAFETY_TITLE = 'Green — safety margin';
export const BAR_SEGMENT_SAFETY_BODY =
  'Part of ctx size left empty on purpose so packing does not assume the full provider window.';

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
