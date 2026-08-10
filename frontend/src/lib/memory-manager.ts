import type {
  MemoryCuratorContext,
  MemoryConsistencyReviewCursor,
  MemoryCuratorMode,
  MemoryRecord,
  MemoryScopeType,
  Project,
  Thread,
} from './api';

export interface MemoryManagerIntent {
  mode: MemoryCuratorMode;
  scopeType: MemoryScopeType;
  scopeId: string;
  threadId?: string | null;
  projectId?: string | null;
  memory?: MemoryRecord | null;
  embeddingModel?: string | null;
  draftContent?: string | null;
}

export const buildCuratorContext = (intent: MemoryManagerIntent): MemoryCuratorContext => ({
  selected_scope_type: intent.scopeType,
  selected_scope_id: intent.scopeType === 'user' ? 'default' : intent.scopeId,
  thread_id: intent.threadId || undefined,
  project_id: intent.projectId || undefined,
});

export const toMemoryConsistencyReviewCursor = (
  review: MemoryConsistencyReviewCursor,
): MemoryConsistencyReviewCursor => ({
  context_type: review.context_type,
  context_id: review.context_id,
  snapshot_at: review.snapshot_at,
  snapshot_scope_versions: review.snapshot_scope_versions,
  anchor_position: review.anchor_position,
  reviewed_anchor_count: review.reviewed_anchor_count,
  remaining_anchor_count: review.remaining_anchor_count,
});

export const createManagerIntent = ({
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
}): MemoryManagerIntent => ({
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

export const defaultMemoryManagerIntent = ({
  thread,
  project,
}: {
  thread?: Thread | null;
  project?: Project | null;
}): MemoryManagerIntent => {
  if (thread) {
    return createManagerIntent({
      scopeType: 'thread',
      scopeId: thread.id,
      thread,
      project,
    });
  }
  if (project) {
    return createManagerIntent({
      scopeType: 'project',
      scopeId: project.id,
      project,
    });
  }
  return createManagerIntent({
    scopeType: 'user',
    scopeId: 'default',
  });
};

export const reviewManagerIntent = (thread: Thread): MemoryManagerIntent => ({
  mode: 'conversation_review',
  scopeType: 'thread',
  scopeId: thread.id,
  threadId: thread.id,
  projectId: thread.project_id,
  ...(thread.embeddingModel ? { embeddingModel: thread.embeddingModel } : {}),
});

export const memoryReviewManagerIntent = ({
  thread,
  project,
}: {
  thread?: Thread | null;
  project?: Project | null;
}): MemoryManagerIntent => ({
  mode: 'memory_review',
  scopeType: thread ? 'thread' : project ? 'project' : 'user',
  scopeId: thread?.id || project?.id || 'default',
  threadId: thread?.id,
  projectId: project?.id || thread?.project_id,
  ...((thread?.embeddingModel || project?.embeddingModel)
    ? { embeddingModel: thread?.embeddingModel || project?.embeddingModel }
    : {}),
});

export const managerTitle = (intent: MemoryManagerIntent) => {
  if (intent.mode === 'conversation_review') return 'Conversation Memory Review';
  if (intent.mode === 'memory_review') return 'Memory Consistency Review';
  if (intent.mode === 'edit') return 'Edit Memory';
  return 'Add Memory';
};
