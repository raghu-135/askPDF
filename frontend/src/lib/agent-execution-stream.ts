export interface AgentExecutionStreamEnvelope {
  id: number | string;
  event: string;
  data: Record<string, any>;
}

type CanonicalAgentRunEvent = {
  id?: number | string;
  event_id?: string;
  sequence?: number;
  attempt?: number;
  kind?: string;
  payload?: Record<string, any>;
  occurred_at?: string;
  terminal?: boolean;
  parallel_groups?: unknown[];
};

async function consumeJsonEventStream(
  response: Response,
  onData: (value: Record<string, any>) => void,
): Promise<void> {
  if (!response.ok) throw new Error(await response.text());
  if (!response.body) throw new Error('The execution stream is unavailable.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      const data = block.split(/\r?\n/).find((line) => line.startsWith('data:'));
      if (!data) continue;
      onData(JSON.parse(data.slice(5).trim()) as Record<string, any>);
    }
    if (done) break;
  }
}

export async function consumeAgentExecutionStream(
  response: Response,
  onEvent: (event: AgentExecutionStreamEnvelope) => void,
): Promise<void> {
  await consumeJsonEventStream(response, (value) => onEvent(value as AgentExecutionStreamEnvelope));
}

export async function consumeCanonicalAgentRunEventStream(
  response: Response,
  onEvent: (event: AgentExecutionStreamEnvelope) => void,
): Promise<void> {
  await consumeJsonEventStream(response, (value) => {
    const event = value as CanonicalAgentRunEvent;
    const payload = event.payload && typeof event.payload === 'object' ? event.payload : {};
    onEvent({
      id: event.id ?? event.sequence ?? 0,
      event: event.kind || 'runtime.event',
      data: {
        ...payload,
        event_id: event.event_id ?? payload.event_id,
        sequence: event.sequence,
        attempt: event.attempt,
        occurred_at: event.occurred_at,
        terminal: event.terminal,
        parallel_groups: event.parallel_groups,
      },
    });
  });
}
