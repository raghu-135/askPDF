/** Normalize terminal agent-run resume SSE payloads into one response shape. */

export type AgentRunResumeStreamResponse = {
  agent_run?: Record<string, unknown>;
  outcome?: Record<string, unknown>;
  answer?: string;
  [key: string]: unknown;
};

function answerFromParts(
  outcome?: Record<string, unknown>,
  agentRun?: Record<string, unknown>,
): string {
  const resultJson = agentRun?.result_json as { answer?: string; final_answer?: string } | undefined;
  return String(
    outcome?.answer
    || outcome?.final_answer
    || resultJson?.answer
    || resultJson?.final_answer
    || '',
  );
}

export function parseAgentRunResumeStreamEvent(
  eventName: string,
  data: Record<string, unknown> | undefined,
): { response: AgentRunResumeStreamResponse; errorMessage?: string } | null {
  if (!data) return null;

  if (eventName === '__error__') {
    const rawError = data.error;
    const errorMessage = String(
      (rawError as { raw_message?: string; message?: string } | undefined)?.raw_message
      || (rawError as { message?: string } | undefined)?.message
      || rawError
      || 'Unable to resume the agent run.',
    );
    return { response: {}, errorMessage };
  }

  let agentRun: Record<string, unknown> | undefined;
  let envelope: Record<string, unknown> = data;

  if (eventName === '__result__' && data.response) {
    envelope = data.response as Record<string, unknown>;
    agentRun = envelope.agent_run as Record<string, unknown> | undefined;
  } else if (data.response) {
    return { response: data.response as AgentRunResumeStreamResponse };
  } else if (data.agent_run) {
    agentRun = data.agent_run as Record<string, unknown>;
    envelope = data;
  } else {
    return null;
  }

  const outcome = envelope.outcome as Record<string, unknown> | undefined;
  return {
    response: {
      ...envelope,
      agent_run: agentRun,
      answer: answerFromParts(outcome, agentRun),
    },
  };
}