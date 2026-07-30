import assert from 'node:assert/strict';
import test from 'node:test';

import {
  LOCAL_USER_MEMORY_SCOPE_ID,
  filterMemoryRecords,
  isMemoryResultTruncated,
  memoryConsentStatus,
  memoryScopesForContext,
  resolveMemoryScopeTarget,
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
  assert.deepEqual(memoryScopesForContext(false), ['user', 'project']);
  assert.deepEqual(memoryScopesForContext(true), ['thread', 'project', 'user']);
});

test('memory filtering searches content, summary, and type', () => {
  const memories = [
    { content: 'Use concise answers', summary: '', memory_type: 'semantic' },
    { content: 'Visited Chicago', summary: 'Travel history', memory_type: 'episodic' },
  ];
  assert.deepEqual(filterMemoryRecords(memories, 'concise'), [memories[0]]);
  assert.deepEqual(filterMemoryRecords(memories, 'travel'), [memories[1]]);
  assert.deepEqual(filterMemoryRecords(memories, 'episodic'), [memories[1]]);
  assert.deepEqual(filterMemoryRecords(memories, '  '), memories);
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
