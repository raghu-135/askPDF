import type {
  MemoryRecord,
  MemoryWorkspaceRecord,
  MemoryWorkspaceSection,
  MemoryScopeType,
  Project,
  Thread,
} from './api';

export const LOCAL_USER_MEMORY_SCOPE_ID = 'default';

export type MemoryScopeTarget = {
  scopeType: MemoryScopeType;
  scopeId: string;
};

export const memoryScopesForContext = (hasThread: boolean): MemoryScopeType[] => (
  hasThread ? ['thread', 'project', 'user'] : ['user']
);

export const filterMemoryRecords = <T extends MemoryRecord>(
  memories: readonly T[],
  query: string,
): T[] => {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return [...memories];
  return memories.filter((memory) => {
    const workspaceMemory = memory as MemoryRecord & Partial<Pick<
      MemoryWorkspaceRecord,
      'applied_overrides' | 'applied_overridden_by'
    >>;
    return memory.content.toLowerCase().includes(normalized)
      || memory.overrides?.some((item) => item.content.toLowerCase().includes(normalized))
      || memory.overridden_by?.some((item) => item.content.toLowerCase().includes(normalized))
      || workspaceMemory.applied_overrides?.some((item) => item.content.toLowerCase().includes(normalized))
      || workspaceMemory.applied_overridden_by?.some((item) => item.content.toLowerCase().includes(normalized));
  });
};

const activityTime = (memory: MemoryRecord): number => {
  const value = memory.updated_at || memory.created_at;
  if (!value) return 0;
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
};

export const sortMemoryRecordsByActivity = <T extends MemoryRecord>(memories: readonly T[]): T[] => (
  [...memories].sort((left, right) => (
    activityTime(right) - activityTime(left) || left.id.localeCompare(right.id)
  ))
);

export const filterMemoryWorkspaceSections = (
  sections: readonly MemoryWorkspaceSection[],
  query: string,
): MemoryWorkspaceSection[] => sections.map((section) => ({
  ...section,
  memories: sortMemoryRecordsByActivity(filterMemoryRecords(section.memories, query)),
}));

export const memorySectionKey = (section: Pick<MemoryWorkspaceSection, 'scope_type' | 'scope_id'>): string => (
  `${section.scope_type}:${section.scope_id}`
);

export const isMemoryResultTruncated = (rowCount: number, limit: number): boolean => (
  limit > 0 && rowCount >= limit
);

export const resolveMemoryScopeTarget = ({
  scopeType,
  thread,
  selectedProjectId,
}: {
  scopeType: MemoryScopeType;
  thread: Thread | null;
  selectedProjectId?: string | null;
}): MemoryScopeTarget | null => {
  if (scopeType === 'user') {
    return { scopeType, scopeId: LOCAL_USER_MEMORY_SCOPE_ID };
  }
  if (scopeType === 'thread') {
    return thread ? { scopeType, scopeId: thread.id } : null;
  }
  const projectId = thread?.project_id || selectedProjectId;
  return projectId ? { scopeType, scopeId: projectId } : null;
};

export const memoryConsentStatus = ({
  scopeType,
  thread,
  project,
}: {
  scopeType: MemoryScopeType;
  thread: Thread | null;
  project?: Project | null;
}): { enabled: boolean | null; label: string } => {
  if (!thread) {
    return { enabled: null, label: 'Administrative access' };
  }
  if (scopeType === 'thread') {
    return { enabled: true, label: 'Used by this thread' };
  }
  if (scopeType === 'project') {
    const enabled = thread.settings?.memory?.thread_reads_project_memory ?? true;
    return {
      enabled,
      label: enabled ? 'Recall enabled' : 'Thread recall disabled',
    };
  }
  const projectEnabled = project?.settings_json?.memory?.project_reads_user_memory === true;
  const threadEnabled = thread.settings?.memory?.thread_reads_user_memory === true;
  const enabled = projectEnabled && threadEnabled;
  return {
    enabled,
    label: enabled
      ? 'Recall enabled'
      : !projectEnabled
        ? 'Project recall disabled'
        : 'Thread recall disabled',
  };
};
