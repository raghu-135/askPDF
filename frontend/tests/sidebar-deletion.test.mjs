import assert from 'node:assert/strict';
import test from 'node:test';

import {
  sidebarDeletionTarget,
  threadsEligibleForProjectDeletion,
} from '../src/lib/sidebar-deletion.ts';

test('Home targets projects while a project workspace targets threads', () => {
  assert.equal(sidebarDeletionTarget(null), 'projects');
  assert.equal(sidebarDeletionTarget('project-1'), 'threads');
});

test('project deletion mode only targets threads from the active project', () => {
  const threads = [
    { id: 'thread-1', project_id: 'project-1' },
    { id: 'thread-2', project_id: 'project-2' },
    { id: 'thread-3', project_id: 'project-1' },
  ];

  assert.deepEqual(
    threadsEligibleForProjectDeletion(threads, 'project-1').map((thread) => thread.id),
    ['thread-1', 'thread-3'],
  );
  assert.deepEqual(threadsEligibleForProjectDeletion(threads, null), []);
});
