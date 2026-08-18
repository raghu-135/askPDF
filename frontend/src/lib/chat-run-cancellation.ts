export interface ActiveChatRun {
  runId?: string;
  running: boolean;
  canceling?: boolean;
}

export function canRequestChatCancellation(run?: ActiveChatRun | null): boolean {
  return Boolean(run?.runId && run.running && !run.canceling);
}

export function recoverCanceledChat<T extends { id: string }>(
  messages: T[],
  optimisticUserId: string,
  optimisticAssistantId: string,
  submittedQuestion: string,
): { messages: T[]; input: string } {
  return {
    messages: messages.filter(
      (message) => message.id !== optimisticUserId && message.id !== optimisticAssistantId,
    ),
    input: submittedQuestion,
  };
}
