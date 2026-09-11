export interface AgentExecutionStreamEnvelope {
  id: number | string;
  event: string;
  data: Record<string, any>;
}

export async function consumeAgentExecutionStream(
  response: Response,
  onEvent: (event: AgentExecutionStreamEnvelope) => void,
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
      onEvent(JSON.parse(data.slice(5).trim()) as AgentExecutionStreamEnvelope);
    }
    if (done) break;
  }
}
