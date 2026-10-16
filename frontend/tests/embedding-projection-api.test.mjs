import assert from 'node:assert/strict';
import test from 'node:test';

function buildEmbeddingProjectionUrl(apiBase, threadId, options) {
  const params = new URLSearchParams();
  if (options?.fileHash) params.set('file_hash', options.fileHash);
  if (options?.sourceKind) params.set('source_kind', options.sourceKind);
  if (options?.sourceFamily) params.set('source_family', options.sourceFamily);
  if (options?.limit != null) params.set('limit', String(options.limit));
  const query = params.toString();
  const base = `${apiBase}/api/threads/${threadId}/embeddings-projection`;
  return query ? `${base}?${query}` : base;
}

test('embedding projection URLs match the frontend client contract', async () => {
  process.env.NEXT_PUBLIC_API_URL = '/api/backend';
  const { resolveApiBase } = await import('../src/lib/api-config.ts');
  const apiBase = resolveApiBase('/api/backend');

  const url = buildEmbeddingProjectionUrl(apiBase, 'thread-1', {
    fileHash: 'fh1',
    limit: 100,
  });
  assert.equal(url, '/api/backend/api/threads/thread-1/embeddings-projection?file_hash=fh1&limit=100');
});

test('embedding projection URLs include source_family', async () => {
  process.env.NEXT_PUBLIC_API_URL = '/api/backend';
  const { resolveApiBase } = await import('../src/lib/api-config.ts');
  const apiBase = resolveApiBase('/api/backend');

  const url = buildEmbeddingProjectionUrl(apiBase, 'thread-1', {
    sourceFamily: 'all',
    limit: 500,
  });
  assert.equal(
    url,
    '/api/backend/api/threads/thread-1/embeddings-projection?source_family=all&limit=500',
  );
});
