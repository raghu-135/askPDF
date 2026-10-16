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

test('SSE reader emits complete events before the stream ends', async () => {
  const encoder = new TextEncoder();
  let sendNext;
  const gate = new Promise((resolve) => { sendNext = resolve; });
  const events = [];
  const response = new Response(new ReadableStream({
    async start(controller) {
      controller.enqueue(encoder.encode('data: {"id":1,"event":"operation.started","data":{"operation_id":"router","run_id":"run-1"}}\n\n'));
      await gate;
      controller.enqueue(encoder.encode('data: {"id":2,"event":"operation.completed","data":{"operation_id":"router","run_id":"run-1"}}\n\n'));
      controller.close();
    },
  }), { status: 200 });

  const done = consumeAgentExecutionStream(response, (event) => events.push(event.event));
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(events, ['operation.started']);
  sendNext();
  await done;
  assert.deepEqual(events, ['operation.started', 'operation.completed']);
});

test('SSE reader rejects unsuccessful responses without retrying', async () => {
  await assert.rejects(
    consumeAgentExecutionStream(new Response('unavailable', { status: 503 }), () => {}),
    /unavailable/,
  );
});
