import assert from 'node:assert/strict';
import test from 'node:test';

import {
  LOCAL_USER_MEMORY_SCOPE_ID,
  filterMemoryRecords,
  filterMemoryWorkspaceSections,
  isMemoryResultTruncated,
  memoryConsentStatus,
  memoryScopesForContext,
  resolveMemoryScopeTarget,
  sortMemoryRecordsByActivity,
} from '../src/lib/memory-workspace.ts';

const project = {
  id: 'project-1',
  settings_json: { memory: { project_reads_user_memory: true } },
};
const thread = {
  id: 'thread-1',
  project_id: 'project-1',
  settings: {
    memory: {
      thread_reads_project_memory: true,
      thread_reads_user_memory: true,
    },
  },
};

test('memory scopes differ between home and thread contexts', () => {
  assert.deepEqual(memoryScopesForContext(false), ['user']);
  assert.deepEqual(memoryScopesForContext(true), ['thread', 'project', 'user']);
});

test('memory filtering searches content and related memories', () => {
  const memories = [
    { content: 'Use concise answers', overrides: [], overridden_by: [] },
    { content: 'Visited Chicago', overrides: [{ content: 'Travel history' }], overridden_by: [] },
  ];
  assert.deepEqual(filterMemoryRecords(memories, 'concise'), [memories[0]]);
  assert.deepEqual(filterMemoryRecords(memories, 'travel'), [memories[1]]);
  assert.deepEqual(filterMemoryRecords(memories, '  '), memories);
});

test('hierarchical sections filter applied relationships without hiding sections', () => {
  const sections = [
    {
      scope_type: 'thread',
      scope_id: 'thread-1',
      recall_enabled: true,
      truncated: false,
      memories: [{
        id: 'newer',
        content: 'Thread preference',
        updated_at: '2026-08-03T12:00:00Z',
        overrides: [],
        overridden_by: [],
        applied_overrides: [{ content: 'Project research focus' }],
        applied_overridden_by: [],
      }],
    },
    {
      scope_type: 'project',
      scope_id: 'project-1',
      recall_enabled: true,
      truncated: false,
      memories: [],
    },
  ];
  const filtered = filterMemoryWorkspaceSections(sections, 'research focus');
  assert.equal(filtered.length, 2);
  assert.equal(filtered[0].memories.length, 1);
  assert.equal(filtered[1].memories.length, 0);
});

test('memory rows sort by update time and then creation time', () => {
  const rows = [
    { id: 'older', content: '', created_at: '2026-08-01T00:00:00Z' },
    { id: 'newer', content: '', created_at: '2026-08-02T00:00:00Z' },
    { id: 'updated', content: '', created_at: '2026-07-01T00:00:00Z', updated_at: '2026-08-03T00:00:00Z' },
  ];
  assert.deepEqual(sortMemoryRecordsByActivity(rows).map((row) => row.id), ['updated', 'newer', 'older']);
});

test('memory result limit reports possible truncation', () => {
  assert.equal(isMemoryResultTruncated(499, 500), false);
  assert.equal(isMemoryResultTruncated(500, 500), true);
  assert.equal(isMemoryResultTruncated(1, 0), false);
});

test('memory scope targets use canonical IDs', () => {
  assert.deepEqual(
    resolveMemoryScopeTarget({ scopeType: 'user', thread: null }),
    { scopeType: 'user', scopeId: LOCAL_USER_MEMORY_SCOPE_ID },
  );
  assert.deepEqual(
    resolveMemoryScopeTarget({ scopeType: 'thread', thread }),
    { scopeType: 'thread', scopeId: 'thread-1' },
  );
  assert.deepEqual(
    resolveMemoryScopeTarget({ scopeType: 'project', thread }),
    { scopeType: 'project', scopeId: 'project-1' },
  );
  assert.equal(
    resolveMemoryScopeTarget({ scopeType: 'project', thread: null, selectedProjectId: '' }),
    null,
  );
});

test('consent status requires both global gates in a thread', () => {
  assert.deepEqual(
    memoryConsentStatus({ scopeType: 'user', thread, project }),
    { enabled: true, label: 'Recall enabled' },
  );
  assert.deepEqual(
    memoryConsentStatus({
      scopeType: 'user',
      thread: {
        ...thread,
        settings: {
          memory: {
            thread_reads_project_memory: true,
            thread_reads_user_memory: false,
          },
        },
      },
      project,
    }),
    { enabled: false, label: 'Thread recall disabled' },
  );
  assert.deepEqual(
    memoryConsentStatus({ scopeType: 'project', thread, project }),
    { enabled: true, label: 'Recall enabled' },
  );
  assert.deepEqual(
    memoryConsentStatus({ scopeType: 'user', thread: null, project }),
    { enabled: null, label: 'Administrative access' },
  );
});
