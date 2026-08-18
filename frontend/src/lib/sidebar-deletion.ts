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
