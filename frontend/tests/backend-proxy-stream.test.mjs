import assert from 'node:assert/strict';
import test from 'node:test';

import { eventStreamProxyHeaders, isEventStreamContentType } from '../src/lib/event-stream-proxy.ts';

test('event-stream proxy headers disable transformation and downstream buffering', () => {
  assert.equal(isEventStreamContentType('text/event-stream'), true);
  assert.equal(isEventStreamContentType('application/json'), false);
  assert.deepEqual(eventStreamProxyHeaders('text/event-stream; charset=utf-8'), {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
});
