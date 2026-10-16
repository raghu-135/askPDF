import { API_BASE } from './api-config';

export type EmbeddingPoint3D = {
  id: string;
  x: number;
  y: number;
  z: number;
  chunk_id?: number | null;
  file_hash: string;
  file_name?: string | null;
  text: string;
  page_start?: number | null;
  page_end?: number | null;
  pages?: string | null;
  source_kind?: string | null;
  table_id?: string | null;
  section_id?: string | null;
};

export type EmbeddingProjectionEdge = {
  id: string;
  source: string;
  target: string;
  label?: string | null;
  kind: 'sequence' | 'similarity' | string;
  score?: number | null;
};

export type EmbeddingProjectionResponse = {
  thread_id: string;
  embedding_model: string;
  point_count: number;
  truncated: boolean;
  points: EmbeddingPoint3D[];
  edges: EmbeddingProjectionEdge[];
};

export type EmbeddingSourceFamily = 'all' | 'documents' | 'chat' | 'web_search' | 'memory';

export async function getThreadEmbeddingProjection(
  threadId: string,
  options?: {
    fileHash?: string;
    sourceKind?: string;
    sourceFamily?: EmbeddingSourceFamily;
    limit?: number;
  },
): Promise<EmbeddingProjectionResponse> {
  const params = new URLSearchParams();
  if (options?.fileHash) params.set('file_hash', options.fileHash);
  if (options?.sourceKind) params.set('source_kind', options.sourceKind);
  if (options?.sourceFamily) params.set('source_family', options.sourceFamily);
  if (options?.limit != null) params.set('limit', String(options.limit));
  const query = params.toString();
  const suffix = query ? `?${query}` : '';
  const response = await fetch(`${API_BASE}/api/threads/${threadId}/embeddings-projection${suffix}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Failed to load embedding projection (${response.status})`);
  }
  return response.json() as Promise<EmbeddingProjectionResponse>;
}
