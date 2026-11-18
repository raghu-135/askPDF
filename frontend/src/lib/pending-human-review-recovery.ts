import { MessageRole } from './enums.ts';

export type HumanReviewRecoveryRun = {
  id: string;
  parent_run_id?: string | null;
  started_at?: string | null;
};

export type ChatMessageForHumanReviewRecovery = {
  id: string;
  role: string;
  content?: string;
  agent_run_id?: string;
  agent_run_turn_kind?: string;
  pending_human_review?: boolean;
  created_at?: string;
};

/** User prompts and persisted turns attach to the parent Auto run when a child delegate pauses. */
export function resolveHumanReviewConversationRunId(run: HumanReviewRecoveryRun): string {
  const parentRunId = String(run.parent_run_id ?? '').trim();
  return parentRunId || run.id;
}

export function hasPersistedUserPromptForRun(
  messages: ChatMessageForHumanReviewRecovery[],
  conversationRunId: string,
): boolean {
  return messages.some(
    (message) =>
      message.role === MessageRole.User
      && String(message.agent_run_id ?? '').trim() === conversationRunId,
  );
}

export function hasRecoveredReviewAssistant(
  messages: ChatMessageForHumanReviewRecovery[],
  childRunId: string,
  localAssistantMessageId: string,
): boolean {
  return messages.some((message) => {
    if (message.id === localAssistantMessageId) return true;
    if (String(message.agent_run_id ?? '').trim() !== childRunId) return false;
    return (
      message.role === MessageRole.Assistant
      && (
        message.agent_run_turn_kind === 'assistant_pending_review'
        || Boolean(message.pending_human_review)
      )
    );
  });
}

export function buildRecoveredPendingHumanReviewMessages(input: {
  run: HumanReviewRecoveryRun;
  interruptTitle: string;
  recoveredQuestion: string;
  localUserMessageId: string;
  localAssistantMessageId: string;
  existingMessages: ChatMessageForHumanReviewRecovery[];
  interruptRequestedAt?: string | null;
}): ChatMessageForHumanReviewRecovery[] | null {
  const childRunId = input.run.id;
  if (
    hasRecoveredReviewAssistant(
      input.existingMessages,
      childRunId,
      input.localAssistantMessageId,
    )
  ) {
    return null;
  }

  const conversationRunId = resolveHumanReviewConversationRunId(input.run);
  const recovered: ChatMessageForHumanReviewRecovery[] = [];
  const question = input.recoveredQuestion.trim();
  if (question && !hasPersistedUserPromptForRun(input.existingMessages, conversationRunId)) {
    recovered.push({
      id: input.localUserMessageId,
      role: MessageRole.User,
      content: question,
      created_at: input.run.started_at || new Date().toISOString(),
      agent_run_id: conversationRunId,
      agent_run_turn_kind: 'user_prompt',
    });
  }

  recovered.push({
    id: input.localAssistantMessageId,
    role: MessageRole.Assistant,
    content: input.interruptTitle || 'Human review required before the agent can continue.',
    created_at: input.interruptRequestedAt || input.run.started_at || new Date().toISOString(),
    agent_run_id: childRunId,
    agent_run_turn_kind: 'assistant_pending_review',
    pending_human_review: true,
  });
  return recovered;
}