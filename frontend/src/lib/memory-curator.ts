import type {
  MemoryCuratorContext,
  MemoryCuratorMode,
  MemoryRecord,
  MemoryScopeType,
  Project,
  Thread,
} from './api';

export interface MemoryCuratorIntent {
  mode: MemoryCuratorMode;
  scopeType: MemoryScopeType;
  scopeId: string;
  threadId?: string | null;
  projectId?: string | null;
  memory?: MemoryRecord | null;
  embeddingModel?: string | null;
}

export const buildCuratorContext = (intent: MemoryCuratorIntent): MemoryCuratorContext => ({
  selected_scope_type: intent.scopeType,
  selected_scope_id: intent.scopeType === 'user' ? 'default' : intent.scopeId,
  thread_id: intent.threadId || undefined,
  project_id: intent.projectId || undefined,
});

export const createCuratorIntent = ({
  scopeType,
  scopeId,
  thread,
  project,
  memory = null,
}: {
  scopeType: MemoryScopeType;
  scopeId: string;
  thread?: Thread | null;
  project?: Project | null;
  memory?: MemoryRecord | null;
}): MemoryCuratorIntent => ({
  mode: memory ? 'edit' : 'create',
  scopeType,
  scopeId: scopeType === 'user' ? 'default' : scopeId,
  threadId: thread?.id,
  projectId: project?.id || thread?.project_id,
  memory,
  ...((memory?.embedding_model || (scopeType !== 'user' && (thread?.embeddingModel || project?.embeddingModel)))
    ? { embeddingModel: memory?.embedding_model || thread?.embeddingModel || project?.embeddingModel }
    : {}),
});

export const defaultMemoryCuratorIntent = ({
  thread,
  project,
}: {
  thread?: Thread | null;
  project?: Project | null;
}): MemoryCuratorIntent => {
  if (thread) {
    return createCuratorIntent({
      scopeType: 'thread',
      scopeId: thread.id,
      thread,
      project,
    });
  }
  if (project) {
    return createCuratorIntent({
      scopeType: 'project',
      scopeId: project.id,
      project,
    });
  }
  return createCuratorIntent({
    scopeType: 'user',
    scopeId: 'default',
  });
};

export const reviewCuratorIntent = (thread: Thread): MemoryCuratorIntent => ({
  mode: 'conversation_review',
  scopeType: 'thread',
  scopeId: thread.id,
  threadId: thread.id,
  projectId: thread.project_id,
  ...(thread.embeddingModel ? { embeddingModel: thread.embeddingModel } : {}),
});

export const memoryReviewCuratorIntent = ({
  thread,
  project,
}: {
  thread?: Thread | null;
  project?: Project | null;
}): MemoryCuratorIntent => ({
  mode: 'memory_review',
  scopeType: thread ? 'thread' : 'project',
  scopeId: thread?.id || project?.id || '',
  threadId: thread?.id,
  projectId: project?.id || thread?.project_id,
  ...((thread?.embeddingModel || project?.embeddingModel)
    ? { embeddingModel: thread?.embeddingModel || project?.embeddingModel }
    : {}),
});

export const curatorTitle = (intent: MemoryCuratorIntent) => {
  if (intent.mode === 'conversation_review') return 'Conversation Memory Review';
  if (intent.mode === 'memory_review') return 'Memory Consistency Review';
  if (intent.mode === 'edit') return 'Edit Memory';
  return 'Add Memory';
};
