/**
 * Live UI / Playwright only — not the product default LLM.
 * Do not use `openrouter/free` (random routing).
 * Verified on OpenRouter model list: supports `tools`, id ends with `:free`.
 */
export const DEFAULT_E2E_LLM_MODEL = 'inclusionai/ling-3.2-flash:free';
export const E2E_LLM_FALLBACK_MODEL = 'openai/gpt-4.1-mini';

export function resolveE2eLlmModel() {
  return process.env.ASKPDF_E2E_LLM_MODEL || DEFAULT_E2E_LLM_MODEL;
}