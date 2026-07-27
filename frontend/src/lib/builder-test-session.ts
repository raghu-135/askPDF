export type BuilderTestMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  runId?: string;
  specFingerprint?: string;
  status?: 'sending' | 'completed' | 'failed' | 'cancelled' | 'review';
};

export type BuilderTestSession = {
  threadId: string | null;
  messages: BuilderTestMessage[];
};

export const emptyBuilderTestSession = (threadId: string | null = null): BuilderTestSession => ({
  threadId,
  messages: [],
});

const stableValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, stableValue(item)]),
    );
  }
  return value;
};

export const workflowSpecFingerprint = (spec: Record<string, unknown>) => {
  const text = JSON.stringify(stableValue(spec));
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `draft-${(hash >>> 0).toString(16).padStart(8, '0')}`;
};

export const transientMessagesForRequest = (messages: BuilderTestMessage[]) => (
  messages
    .filter((message) => message.status === 'completed' && message.content.trim())
    .map(({ role, content }) => ({ role, content }))
);
