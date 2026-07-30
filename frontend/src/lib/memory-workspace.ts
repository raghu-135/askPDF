import type {
  MemoryRecord,
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
  hasThread ? ['thread', 'project', 'user'] : ['user', 'project']
);

export const filterMemoryRecords = (
  memories: readonly MemoryRecord[],
  query: string,
): MemoryRecord[] => {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return [...memories];
  return memories.filter((memory) => (
    memory.content.toLowerCase().includes(normalized)
    || String(memory.summary || '').toLowerCase().includes(normalized)
    || memory.memory_type.toLowerCase().includes(normalized)
  ));
};

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
