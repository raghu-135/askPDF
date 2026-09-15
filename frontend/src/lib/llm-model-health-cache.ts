export type LlmModelHealth = {
  ready: boolean;
  supportsTools: boolean;
  canInvokeTools: boolean;
};

const llmHealthCache = new Map<string, LlmModelHealth>();
const llmHealthInflight = new Map<string, Promise<LlmModelHealth>>();

export const peekLlmModelHealth = (model: string): LlmModelHealth | null => (
  llmHealthCache.get(model) ?? null
);

export const clearLlmModelHealthCache = (): void => {
  llmHealthCache.clear();
  llmHealthInflight.clear();
};

export const loadLlmModelHealth = async (
  model: string,
  loader: (model: string) => Promise<LlmModelHealth>,
): Promise<LlmModelHealth> => {
  const cached = llmHealthCache.get(model);
  if (cached) return cached;
  const inflight = llmHealthInflight.get(model);
  if (inflight) return inflight;

  const request = (async (): Promise<LlmModelHealth> => {
    try {
      const result = await loader(model);
      llmHealthCache.set(model, result);
      return result;
    } finally {
      llmHealthInflight.delete(model);
    }
  })();
  llmHealthInflight.set(model, request);
  return request;
};
