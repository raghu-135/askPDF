import assert from 'node:assert/strict';
import test from 'node:test';

import { consumeAgentExecutionStream, consumeCanonicalAgentRunEventStream } from '../src/lib/agent-execution-stream.ts';

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

test('canonical run-event SSE is adapted into a live trace envelope', async () => {
  const encoder = new TextEncoder();
  const response = new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode([
        'id: 7',
        'event: run_event',
        'data: {"id":"row-7","event_id":"event-7","sequence":7,"attempt":2,"kind":"operation.started","payload":{"operation_id":"planner"},"occurred_at":"2026-09-11T10:00:00Z","terminal":false,"parallel_groups":[]}',
        '',
        '',
      ].join('\n')));
      controller.close();
    },
  }), { status: 200 });
  const events = [];

  await consumeCanonicalAgentRunEventStream(response, (event) => events.push(event));

  assert.deepEqual(events, [{
    id: 'row-7',
    event: 'operation.started',
    data: {
      operation_id: 'planner',
      event_id: 'event-7',
      sequence: 7,
      attempt: 2,
      occurred_at: '2026-09-11T10:00:00Z',
      terminal: false,
      parallel_groups: [],
    },
  }]);
});
