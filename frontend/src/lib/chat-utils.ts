// Centralized chat and thread utilities for frontend
// Place all shared logic here for use in ChatInterface and ThreadSidebar

/**
 * Formats a date string into a human-readable label (Today, Yesterday, X days ago, or locale date).
 * @param dateStr - The date string to format.
 * @returns A formatted date string.
 */
export const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return date.toLocaleDateString();
};

/**
 * Fetches available embedding models from the backend RAG API.
 * @param ragApiUrl - The base URL of the RAG API.
 * @returns A promise resolving to an array of embedding model names.
 */
export const fetchAvailableEmbedModels = async (ragApiUrl: string): Promise<string[]> => {
  try {
    const res = await fetch(`${ragApiUrl}/models`);
    const data = await res.json();
    if (data.embedding_models || data.not_embedding_models) {
      return [...(data.embedding_models || []), ...(data.not_embedding_models || [])];
    } else if (data.all_models) {
      // If all_models is an array of objects, extract id or return as is
      if (Array.isArray(data.all_models) && data.all_models.length > 0 && typeof data.all_models[0] === 'object' && 'id' in data.all_models[0]) {
        return data.all_models.map((m: any) => m.id);
      }
      return data.all_models;
    } else {
      return [];
    }
  } catch (err) {
    console.warn("Failed to fetch embedding models", err);
    return [];
  }
};

/**
 * Fetches available LLM models from the backend RAG API.
 * @param ragApiUrl - The base URL of the RAG API.
 * @returns A promise resolving to an array of LLM model names.
 */
export const fetchAvailableLlmModels = async (ragApiUrl: string): Promise<string[]> => {
  try {
    const res = await fetch(`${ragApiUrl}/models`);
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
 * Checks if the specified embedding model is ready on the backend.
 * @param model - The embedding model name to check.
 * @param apiBase - The base URL of the backend API.
 * @returns A promise resolving to true if the model is ready, false otherwise.
 */
export const checkEmbedModelReady = async (model: string, apiBase: string): Promise<boolean> => {
  try {
    const res = await fetch(`${apiBase}/health/is_embed_model_ready?model=${encodeURIComponent(model)}`);
    const data = await res.json();
    return data.embed_model_ready === true;
  } catch {
    return false;
  }
};

/**
 * Checks if the specified LLM (chat) model is ready on the backend.
 * @param model - The LLM model name to check.
 * @param apiBase - The base URL of the backend API.
 * @returns A promise resolving to true if the model is ready, false otherwise.
 */
export const checkLlmModelReady = async (model: string, apiBase: string): Promise<boolean> => {
  try {
    const res = await fetch(`${apiBase}/health/is_chat_model_ready?model=${encodeURIComponent(model)}`);
    const data = await res.json();
    return data.ready === true || data.chat_model_ready === true;
  } catch {
    return false;
  }
};
