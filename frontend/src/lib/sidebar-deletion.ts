import { activityTimestamp, sortByRecentActivity } from './recency.ts';

export type SidebarDeletionTarget = 'projects' | 'threads';

export function sidebarDeletionTarget(
  activeProjectId?: string | null,
): SidebarDeletionTarget {
  return activeProjectId ? 'threads' : 'projects';
}

export function threadsEligibleForProjectDeletion<T extends { project_id?: string | null }>(
  threads: readonly T[],
  activeProjectId?: string | null,
): T[] {
  if (!activeProjectId) return [];
  return threads.filter((thread) => thread.project_id === activeProjectId);
}

export function sidebarGroupsForProject<T extends { project?: { id?: string } | null }>(
  groups: readonly T[],
  activeProjectId?: string | null,
): T[] {
  if (!activeProjectId) return [...groups];
  return groups.filter((group) => group.project?.id === activeProjectId);
}

export function sortSidebarGroupsByActivity<
  T extends {
    project?: {
      last_activity_at?: string | null;
      updated_at?: string | null;
      created_at?: string | null;
    } | null;
    threads: Array<{
      last_activity_at?: string | null;
      updated_at?: string | null;
      created_at?: string | null;
    }>;
  },
>(groups: readonly T[]): T[] {
  return [...groups]
    .map((group) => ({
      ...group,
      threads: sortByRecentActivity(group.threads),
    }))
    .sort((left, right) => {
      const leftActivity = Math.max(
        activityTimestamp(left.project || {}),
        ...left.threads.map((thread) => activityTimestamp(thread)),
        0,
      );
      const rightActivity = Math.max(
        activityTimestamp(right.project || {}),
        ...right.threads.map((thread) => activityTimestamp(thread)),
        0,
      );
      return rightActivity - leftActivity;
    });
}
