import assert from 'node:assert/strict';
import test from 'node:test';

function buildFileChunksUrl(
  apiBase,
  target,
  fileHash,
  options,
) {
  const params = new URLSearchParams();
  if (options?.limit != null) params.set('limit', String(options.limit));
  if (options?.offset != null) params.set('offset', String(options.offset));
  const query = params.toString();
  const base =
    target.scope === 'thread'
      ? `${apiBase}/api/threads/${target.id}/files/${fileHash}/chunks`
      : `${apiBase}/api/projects/${target.id}/files/${fileHash}/chunks`;
  return query ? `${base}?${query}` : base;
}

test('file chunk inspection URLs match the frontend client contract', async () => {
  process.env.NEXT_PUBLIC_API_URL = '/api/backend';
  const { resolveApiBase } = await import('../src/lib/api-config.ts');
  const apiBase = resolveApiBase('/api/backend');

  const threadUrl = buildFileChunksUrl(
    apiBase,
    { scope: 'thread', id: 'thread-1' },
    'fh1',
    { limit: 25, offset: 50 },
  );
  assert.equal(threadUrl, '/api/backend/api/threads/thread-1/files/fh1/chunks?limit=25&offset=50');

  const projectUrl = buildFileChunksUrl(
    apiBase,
    { scope: 'project', id: 'project-9' },
    'fh2',
  );
  assert.equal(projectUrl, '/api/backend/api/projects/project-9/files/fh2/chunks');
});
