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
    const nested = data.response as Record<string, unknown>;
    const nestedRun = (nested.agent_run as Record<string, unknown> | undefined)
      ?? (data.agent_run as Record<string, unknown> | undefined)
      ?? (nested.pending_interrupt
        ? {
            status: nested.status ?? data.status ?? 'awaiting_human',
            pending_interrupt: nested.pending_interrupt,
          }
        : undefined);
    const nestedOutcome = nested.outcome as Record<string, unknown> | undefined;
    return {
      response: {
        ...nested,
        agent_run: nestedRun,
        answer: answerFromParts(nestedOutcome, nestedRun),
      },
    };
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

/** True when an SSE event carries the product-level resume result (not a raw runtime terminal). */
export function isAgentRunResumeProductTerminal(
  eventName: string,
  data: Record<string, unknown> | undefined,
): boolean {
  if (!data) return false;
  if (eventName === '__result__' || eventName === '__error__') return true;
  if (eventName === 'stream.error' || eventName === 'run.failed' || eventName === 'run.cancelled') return true;
  if (eventName === 'run.completed' || eventName === 'interrupt.requested') {
    const agentRun = (data.agent_run ?? (data.response as Record<string, unknown> | undefined)?.agent_run) as
      | Record<string, unknown>
      | undefined;
    if (agentRun && typeof agentRun === 'object') {
      return true;
    }
    return Object.prototype.hasOwnProperty.call(data, 'outcome');
  }
  return false;
}