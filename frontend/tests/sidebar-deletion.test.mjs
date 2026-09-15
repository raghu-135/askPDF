import assert from 'node:assert/strict';
import test from 'node:test';

import {
  sidebarDeletionTarget,
  sidebarGroupsForProject,
  sortSidebarGroupsByActivity,
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

test('an open project hides other sidebar groups', () => {
  const groups = [
    { project: { id: 'project-1' } },
    { project: { id: 'project-2' } },
  ];
  assert.deepEqual(
    sidebarGroupsForProject(groups, 'project-2').map((group) => group.project?.id),
    ['project-2'],
  );
  assert.equal(sidebarGroupsForProject(groups, null).length, 2);
});

test('sidebar groups and threads sort by recent activity', () => {
  const groups = [
    {
      project: { id: 'older', last_activity_at: '2026-09-01T00:00:00.000Z' },
      threads: [
        { created_at: '2026-09-02T00:00:00.000Z' },
        { last_activity_at: '2026-09-10T00:00:00.000Z' },
      ],
    },
    {
      project: { id: 'newer', last_activity_at: '2026-09-12T00:00:00.000Z' },
      threads: [{ created_at: '2026-09-03T00:00:00.000Z' }],
    },
  ];
  const sorted = sortSidebarGroupsByActivity(groups);
  assert.deepEqual(sorted.map((group) => group.project?.id), ['newer', 'older']);
  assert.equal(sorted[1].threads[0].last_activity_at, '2026-09-10T00:00:00.000Z');
});
