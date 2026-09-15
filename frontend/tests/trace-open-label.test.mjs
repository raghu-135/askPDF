import assert from 'node:assert/strict';
import test from 'node:test';

import { formatAgentTraceIdentity, formatOpenTraceControlLabel } from '../src/lib/trace-open-label.ts';

test('trace identity uses the workflow id and optional route', () => {
  assert.equal(formatAgentTraceIdentity({}), 'agent');
  assert.equal(formatAgentTraceIdentity({ workflowId: '  ' }), 'agent');
  assert.equal(formatAgentTraceIdentity({ workflowId: 'router_rag_agent' }), 'router_rag_agent');
  assert.equal(
    formatAgentTraceIdentity({ workflowId: 'router_rag_agent', route: 'document' }),
    'router_rag_agent · document',
  );
});

test('live open-trace control shows the running workflow before the answer', () => {
  assert.equal(
    formatOpenTraceControlLabel({ running: true, workflowId: 'router_rag_agent' }),
    'Open live trace · router_rag_agent',
  );
  assert.equal(
    formatOpenTraceControlLabel({
      running: true,
      workflowId: 'router_rag_agent',
      route: 'document',
    }),
    'Open live trace · router_rag_agent · document',
  );
  assert.equal(
    formatOpenTraceControlLabel({
      running: false,
      workflowId: 'router_rag_agent',
      route: 'document',
    }),
    'Open trace · router_rag_agent · document',
  );
  assert.equal(
    formatOpenTraceControlLabel({ canceling: true, running: true, workflowId: 'router_rag_agent' }),
    'Stopping after current step…',
  );
});
