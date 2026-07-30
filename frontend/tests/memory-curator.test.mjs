import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildCuratorContext,
  createCuratorIntent,
  curatorTitle,
  reviewCuratorIntent,
} from '../src/lib/memory-curator.ts';

const project = { id: 'project-1' };
const thread = { id: 'thread-1', project_id: 'project-1' };

test('curator intents preserve workspace scope and canonical global ID', () => {
  const intent = createCuratorIntent({
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

test('selected memory opens edit mode and conversation review stays thread-scoped', () => {
  const memory = { id: 'memory-1', content: 'Use concise answers.' };
  const edit = createCuratorIntent({
    scopeType: 'project',
    scopeId: project.id,
    thread,
    project,
    memory,
  });
  const review = reviewCuratorIntent(thread);

  assert.equal(edit.mode, 'edit');
  assert.equal(edit.memory, memory);
  assert.equal(curatorTitle(edit), 'Edit Memory');
  assert.deepEqual(review, {
    mode: 'conversation_review',
    scopeType: 'thread',
    scopeId: 'thread-1',
    threadId: 'thread-1',
    projectId: 'project-1',
  });
});
