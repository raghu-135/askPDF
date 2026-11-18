import { MessageRole } from './enums.ts';

const TOOL_APPROVAL_TITLE = /^Approve\s+.+\?$/i;

export function isToolApprovalPlaceholderContent(content: string): boolean {
  return TOOL_APPROVAL_TITLE.test(String(content || '').trim());
}

type InterruptPlaceholderMessage = {
  id: string;
  role: string;
  content: string;
  agent_run_id?: string;
  agent_run_turn_kind?: string;
  pending_human_review?: boolean;
};

export function isChatInterruptPlaceholderMessage(message: InterruptPlaceholderMessage): boolean {
  if (message.role !== MessageRole.Assistant) return false;
  return (
    message.agent_run_turn_kind === 'assistant_pending_review'
    || Boolean(message.pending_human_review)
    || (Boolean(message.agent_run_id) && isToolApprovalPlaceholderContent(message.content))
  );
}

/** Hide interrupt placeholder bubbles; the decision panel owns the approval UX. */
export function shouldHideChatInterruptPlaceholder(
  message: InterruptPlaceholderMessage,
): boolean {
  return isChatInterruptPlaceholderMessage(message);
}

/** After HITL approval, show the assistant row like a normal in-flight run (trace + stop). */
export function clearInterruptPlaceholderForLiveRun<T extends InterruptPlaceholderMessage>(message: T): T {
  if (!isChatInterruptPlaceholderMessage(message)) return message;
  const placeholderContent = isToolApprovalPlaceholderContent(message.content)
    || message.agent_run_turn_kind === 'assistant_pending_review'
    || Boolean(message.pending_human_review);
  return {
    ...message,
    pending_human_review: false,
    agent_run_turn_kind: 'assistant_final',
    content: placeholderContent ? '' : message.content,
  };
}