import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildCuratorContext,
  createManagerIntent,
  managerTitle,
  defaultMemoryManagerIntent,
  memoryReviewManagerIntent,
  reviewManagerIntent,
  toMemoryConsistencyReviewCursor,
} from '../src/lib/memory-manager.ts';

const project = { id: 'project-1' };
const thread = { id: 'thread-1', project_id: 'project-1' };

test('default curator intent follows the active workspace context', () => {
  assert.deepEqual(defaultMemoryManagerIntent({}), {
    mode: 'create',
    scopeType: 'user',
    scopeId: 'default',
    threadId: undefined,
    projectId: undefined,
    memory: null,
  });
  assert.equal(defaultMemoryManagerIntent({ project }).scopeType, 'project');
  assert.equal(defaultMemoryManagerIntent({ thread, project }).scopeType, 'thread');

  const readyProject = { ...project, embeddingModel: 'project-model' };
  assert.equal(
    defaultMemoryManagerIntent({ project: readyProject }).embeddingModel,
    'project-model',
  );
});

test('review cursor serialization excludes transient candidate metadata', () => {
  assert.deepEqual(toMemoryConsistencyReviewCursor({
    context_type: 'thread',
    context_id: 'thread-1',
    snapshot_at: '2026-08-03T00:00:00Z',
    snapshot_scope_versions: { 'thread:thread-1': 2 },
    anchor_position: 5,
    reviewed_anchor_count: 5,
    remaining_anchor_count: 1,
    candidate_groups: [{ anchor_id: 'memory-1' }],
    representation_pending: false,
  }), {
    context_type: 'thread',
    context_id: 'thread-1',
    snapshot_at: '2026-08-03T00:00:00Z',
    snapshot_scope_versions: { 'thread:thread-1': 2 },
    anchor_position: 5,
    reviewed_anchor_count: 5,
    remaining_anchor_count: 1,
  });
});

test('curator intents preserve workspace scope and canonical global ID', () => {
  const intent = createManagerIntent({
    scopeType: 'user',
    scopeId: 'legacy-user',
    thread,
    project,
  });

  assert.equal(intent.scopeId, 'default');
  assert.deepEqual(buildCuratorContext(intent), {
    selected_scope_type: 'user',
    selected_scope_id: 'default',
    thread_id: 'thread-1',
    project_id: 'project-1',
  });
});

test('consistency review binds to the active project or thread context', () => {
  const globalReview = memoryReviewManagerIntent({});
  const projectReview = memoryReviewManagerIntent({ project });
  const threadReview = memoryReviewManagerIntent({ thread, project });

  assert.deepEqual(globalReview, {
    mode: 'memory_review',
    scopeType: 'user',
    scopeId: 'default',
    threadId: undefined,
    projectId: undefined,
  });
  assert.deepEqual(projectReview, {
    mode: 'memory_review',
    scopeType: 'project',
    scopeId: 'project-1',
    threadId: undefined,
    projectId: 'project-1',
  });
  assert.equal(threadReview.mode, 'memory_review');
  assert.equal(threadReview.scopeType, 'thread');
  assert.equal(threadReview.scopeId, 'thread-1');
  assert.equal(managerTitle(threadReview), 'Memory Consistency Review');
});

test('selected memory opens edit mode and conversation review stays thread-scoped', () => {
  const memory = { id: 'memory-1', content: 'Use concise answers.' };
  const edit = createManagerIntent({
    scopeType: 'project',
    scopeId: project.id,
    thread,
    project,
    memory,
  });
  const review = reviewManagerIntent(thread);

  assert.equal(edit.mode, 'edit');
  assert.equal(edit.memory, memory);
  assert.equal(managerTitle(edit), 'Edit Memory');
  assert.deepEqual(review, {
    mode: 'conversation_review',
    scopeType: 'thread',
    scopeId: 'thread-1',
    threadId: 'thread-1',
    projectId: 'project-1',
  });
});
