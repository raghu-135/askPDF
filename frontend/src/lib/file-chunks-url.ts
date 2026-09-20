import { API_BASE } from './api-config';

export interface VectorChunk {
  chunk_id: number;
  source_id?: string | null;
  chunk_identity?: string | null;
  text: string;
  file_hash?: string | null;
  manifest_id?: string | null;
  generation?: string | null;
  section_id?: string | null;
  table_id?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  pages?: string | null;
  source_kind?: string | null;
  url?: string | null;
  title?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface FileChunksResponse {
  file_hash: string;
  embedding_model: string;
  total_count: number;
  limit: number;
  offset: number;
  chunks: VectorChunk[];
}

export function buildFileChunksUrl(
  target: { scope: 'thread'; id: string } | { scope: 'project'; id: string },
  fileHash: string,
  options?: { limit?: number; offset?: number },
): string {
  const params = new URLSearchParams();
  if (options?.limit != null) params.set('limit', String(options.limit));
  if (options?.offset != null) params.set('offset', String(options.offset));
  const query = params.toString();
  const base =
    target.scope === 'thread'
      ? `${API_BASE}/api/threads/${target.id}/files/${fileHash}/chunks`
      : `${API_BASE}/api/projects/${target.id}/files/${fileHash}/chunks`;
  return query ? `${base}?${query}` : base;
}

export async function getFileChunks(
  target: { scope: 'thread'; id: string } | { scope: 'project'; id: string },
  fileHash: string,
  options?: { limit?: number; offset?: number },
): Promise<FileChunksResponse> {
  const res = await fetch(buildFileChunksUrl(target, fileHash, options));
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
