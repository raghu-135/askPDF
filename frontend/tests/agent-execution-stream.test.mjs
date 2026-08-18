import assert from 'node:assert/strict';
import test from 'node:test';

import { consumeAgentExecutionStream } from '../src/lib/agent-execution-stream.ts';

test('SSE reader handles fragmented blocks and preserves event envelopes', async () => {
  const encoder = new TextEncoder();
  const chunks = [
    'id: 1\nevent: run.started\ndata: {"id":1,"event":"run.started","data":{"run_id":"run-1"}}\n',
    '\nid: 2\nevent: node.completed\ndata: {"id":2,"event":"node.completed","data":{"node_id":"router","visit_index":1}}\n\n',
  ];
  const response = new Response(new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  }), { status: 200 });
  const events = [];

  await consumeAgentExecutionStream(response, (event) => events.push(event));

  assert.deepEqual(events.map((event) => event.event), ['run.started', 'node.completed']);
  assert.equal(events[1].data.node_id, 'router');
});

test('SSE reader rejects unsuccessful responses without retrying', async () => {
  await assert.rejects(
    consumeAgentExecutionStream(new Response('unavailable', { status: 503 }), () => {}),
    /unavailable/,
  );
});
