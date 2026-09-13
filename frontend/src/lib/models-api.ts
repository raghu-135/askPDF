/**
 * Model-related API calls for fetching and checking model health/status.
 */

import { API_BASE } from "./api";
import ErrorIcon from '@mui/icons-material/Error';

/**
 * Fetches available embedding models from the backend RAG API.
 * @returns A promise resolving to a map with embedding model categories.
 */
export const fetchAvailableEmbeddingModels = async (): Promise<{
  embedding_models: string[];
  local_embedding_models: string[];
  not_embedding_models: string[];
}> => {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    const data = await res.json();
    return {
      embedding_models: data.embedding_models || [],
      local_embedding_models: data.local_embedding_models || [],
      not_embedding_models: data.not_embedding_models || [],
    };
  } catch (err) {
    console.warn("Failed to fetch embedding models", err);
    return {
      embedding_models: [],
      local_embedding_models: [],
      not_embedding_models: [],
    };
  }
};

/**
 * Fetches available LLM models from the backend RAG API.
 * @returns A promise resolving to an array of LLM model names.
 */
export const fetchAvailableLlmModels = async (): Promise<string[]> => {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    const data = await res.json();
    if (data.llm_models || data.not_llm_models) {
      return [...(data.llm_models || []), ...(data.not_llm_models || [])];
    } else if (data.all_models && data.all_models.length > 0) {
      return data.all_models.map((m: any) => m.id);
    } else {
      throw new Error("No models found");
    }
  } catch (err) {
    console.error("Failed to fetch models", err);
    return [];
  }
};

/**
 * Fetches available LLM candidates while preserving discovery failures for
 * controls that need distinct loading, empty, and error states.
 */
export const fetchAvailableLlmModelsStrict = async (): Promise<string[]> => {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error(`Unable to load models (${res.status}).`);
  const data = await res.json();
  const candidates = data.llm_models || data.not_llm_models
    ? [...(data.llm_models || []), ...(data.not_llm_models || [])]
    : Array.isArray(data.all_models)
      ? data.all_models.map((model: any) => model?.id || model).filter(Boolean)
      : [];
  return Array.from(new Set(candidates.map(String)));
};

/**
 * Checks if the specified embedding model is ready on the backend.
 * @param model - The embedding model name to check.
 * @returns A promise resolving to true if the model is ready, false otherwise.
 */
export const checkEmbeddingModelReady = async (model: string): Promise<boolean> => {
  try {
    const res = await fetch(`${API_BASE}/api/health/embed-model/${encodeURIComponent(model)}`);
    const data = await res.json();
    return data.embedding_model_ready === true;
  } catch {
    return false;
  }
};

/**
 * Checks if the specified LLM (chat) model is ready and supports tool calling.
 * @param model - The LLM model name to check.
 * @returns A promise resolving to { ready: boolean, supportsTools: boolean }.
 */
export const checkLlmModelReady = async (
  model: string
): Promise<{ ready: boolean; supportsTools: boolean; canInvokeTools: boolean }> => {
  try {
    const res = await fetch(`${API_BASE}/api/health/chat-model/${encodeURIComponent(model)}`);
    const data = await res.json();
    return {
      ready: data.ready === true || data.chat_model_ready === true,
      supportsTools: data.supports_tools === true,
      canInvokeTools: data.can_invoke_tools === true,
    };
  } catch {
    return { ready: false, supportsTools: false, canInvokeTools: false };
  }
};
