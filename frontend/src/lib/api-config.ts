export function resolveApiBase(
  configuredUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL,
): string {
  const normalized = configuredUrl?.trim();
  if (normalized) return normalized.replace(/\/+$/, '');
  throw new Error(
    'NEXT_PUBLIC_API_URL is required. Set it to the public RAG service URL before starting or building the frontend.',
  );
}

export const API_BASE = resolveApiBase();

export function buildAgentWorkflowCatalogUrl(
  apiBase: string,
  framework: string,
  builderId: string,
): string {
  const params = new URLSearchParams({ framework, builder_id: builderId });
  return `${apiBase}/api/internal/agent-workflows/catalog?${params.toString()}`;
}
