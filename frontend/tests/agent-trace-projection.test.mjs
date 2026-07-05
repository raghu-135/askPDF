import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildRunGraphOverlay,
  buildRunTraceView,
} from '../src/components/agent-debug/agent-trace-projection.ts';

const resolvedSpec = {
  config: {
    graph: {
      nodes: [
        { id: 'context_loader', type: 'context_loader' },
        { id: 'planner', type: 'planner' },
        { id: 'retrieval_worker', type: 'retrieval_worker' },
        { id: 'memory_worker', type: 'memory_worker' },
      ],
      edges: [],
    },
    allowed_tool_ids: ['document_evidence', 'deep_memory'],
  },
};

const traceBackedRun = {
  id: 'run-1',
  template_id: 'plan_execute_rag_agent',
  resolved_spec_json: resolvedSpec,
  debug: {
    route: 'legacy-route',
    metrics: {
      duration_ms: 25,
      tool_event_count: 1,
      tool_warning_count: 0,
      error_count: 0,
    },
    node_events: [{ node: 'legacy_router', elapsed_ms: 99 }],
    tool_events: [{ tool_name: 'legacy_tool', caller_node: 'legacy_router' }],
    trace: {
      schema_version: 1,
      trace_id: 'run-1',
      run_id: 'run-1',
      template_id: 'plan_execute_rag_agent',
      pattern_type: 'plan_execute_rag_agent',
      status: 'completed',
      attributes: {
        'askpdf.route': 'execute',
        'askpdf.route_reason': 'Document evidence requested.',
      },
      metrics: {
        duration_ms: 25,
        tool_event_count: 1,
        tool_warning_count: 0,
        error_count: 0,
      },
      spans: [
        {
          span_id: 'run:run-1',
          parent_span_id: null,
          name: 'Agent Run',
          kind: 'AGENT',
          status: 'completed',
          attributes: {},
        },
        {
          span_id: 'node:planner:0',
          parent_span_id: 'run:run-1',
          name: 'Planner',
          kind: 'AGENT',
          status: 'completed',
          duration_ms: 5,
          attributes: {
            'askpdf.node.id': 'planner',
            'askpdf.route': 'execute',
            'askpdf.route_reason': 'Document evidence requested.',
            'askpdf.execution_plan': ['retrieval_worker'],
          },
          input: { value: { question: 'Find source' } },
          output: { value: { route: 'execute' } },
          events: [
            {
              name: 'decision.made',
              attributes: {
                'askpdf.route': 'execute',
                'askpdf.route_reason': 'Document evidence requested.',
                'askpdf.execution_plan': ['retrieval_worker'],
              },
            },
            {
              name: 'prompt.rendered',
              attributes: {
                'prompt.name': 'Planner Node Prompt',
                'prompt.chars': 1234,
              },
              output: {
                system_message: 'You are a strict planner.',
                preview: 'Plan safely.',
              },
            },
          ],
          raw: {
            node: 'planner',
            route: 'execute',
            route_reason: 'Document evidence requested.',
            execution_plan: ['retrieval_worker'],
            elapsed_ms: 5,
          },
        },
        {
          span_id: 'node:retrieval_worker:1',
          parent_span_id: 'run:run-1',
          name: 'Document Retrieval',
          kind: 'RETRIEVER',
          status: 'completed',
          duration_ms: 8,
          attributes: {
            'askpdf.node.id': 'retrieval_worker',
          },
          output: {
            value: 'Found document evidence.',
            refs: {
              document_refs: [
                {
                  file_hash: 'abc',
                  file_name: 'paper.pdf',
                  chunk_id: 'chunk-1',
                  page_start: 2,
                },
              ],
            },
          },
          events: [],
          raw: {
            node: 'retrieval_worker',
            elapsed_ms: 8,
            output_preview: 'Found document evidence.',
          },
        },
        {
          span_id: 'node:memory_worker:2',
          parent_span_id: 'run:run-1',
          name: 'Memory Retrieval',
          kind: 'RETRIEVER',
          status: 'skipped',
          duration_ms: 0.5,
          attributes: {
            'askpdf.node.id': 'memory_worker',
            'askpdf.skip_reason': 'not_selected_by_plan',
          },
          events: [
            {
              name: 'skipped',
              attributes: { 'askpdf.skip_reason': 'not_selected_by_plan' },
            },
          ],
          raw: {
            node: 'memory_worker',
            skipped: true,
            status: 'skipped',
            skip_reason: 'not_selected_by_plan',
            elapsed_ms: 0.5,
          },
        },
        {
          span_id: 'tool:search_documents:0',
          parent_span_id: 'node:retrieval_worker:1',
          name: 'Document Evidence',
          kind: 'RETRIEVER',
          status: 'completed',
          duration_ms: 7,
          attributes: {
            'tool.name': 'search_documents',
            'tool.id': 'document_evidence',
            'askpdf.tool.category': 'document',
            'askpdf.caller_node': 'retrieval_worker',
            'askpdf.source_count': 1,
          },
          input: { value: { query: 'Find source' } },
          output: {
            value: 'Found document evidence.',
            refs: {
              document_refs: [{ file_hash: 'abc', chunk_id: 'chunk-1' }],
            },
            summary: { document_refs: 1 },
          },
          events: [
            {
              name: 'tool.completed',
              attributes: {
                'tool.name': 'search_documents',
                'askpdf.source_count': 1,
              },
            },
          ],
          raw: {
            tool_name: 'search_documents',
            tool_id: 'document_evidence',
            caller_node: 'retrieval_worker',
            ok: true,
            elapsed_ms: 7,
            source_count: 1,
          },
        },
      ],
      raw: {
        node_events: [],
        tool_events: [],
      },
    },
  },
};

test('trace projection prefers debug.trace over legacy node and tool events', () => {
  const view = buildRunTraceView(traceBackedRun);
  const overlay = buildRunGraphOverlay(view);

  assert.equal(view.route, 'execute');
  assert.equal(view.routeReason, 'Document evidence requested.');
  assert.deepEqual(view.nodes.map((node) => node.id), ['planner', 'retrieval_worker', 'memory_worker']);
  assert.deepEqual(view.tools.map((tool) => tool.name), ['search_documents']);
  assert.equal(view.nodes[0].span?.span_id, 'node:planner:0');
  assert.equal(view.tools[0].span?.span_id, 'tool:search_documents:0');
  assert.equal(overlay.route, 'execute');
  assert.equal(overlay.routeReason, 'Document evidence requested.');
  assert.deepEqual(overlay.nodeEvents?.map((event) => event.node), ['planner', 'retrieval_worker', 'memory_worker']);
  assert.deepEqual(overlay.toolEvents?.map((event) => event.tool_name), ['search_documents']);
});

test('trace summary counts non-skipped nodes and available graph nodes', () => {
  const view = buildRunTraceView(traceBackedRun);

  assert.equal(view.usedNodeCount, 2);
  assert.equal(view.availableNodeCount, 4);
  assert.equal(view.usedToolCount, 1);
  assert.equal(view.availableToolCount, 2);
  assert.equal(view.warningCount, 0);
  assert.equal(view.errorCount, 0);
});

test('legacy debug projection still works without debug.trace', () => {
  const legacyRun = {
    id: 'run-legacy',
    template_id: 'router_rag_agent',
    resolved_spec_json: resolvedSpec,
    debug: {
      route: 'document',
      node_events: [
        { node: 'router', route: 'document', elapsed_ms: 3 },
        { node: 'retrieval_worker', elapsed_ms: 8 },
      ],
      tool_events: [
        { tool_name: 'search_documents', caller_node: 'retrieval_worker', ok: true },
      ],
      tool_event_count: 1,
      tool_warning_count: 0,
      tool_error_count: 0,
    },
  };

  const view = buildRunTraceView(legacyRun);
  const overlay = buildRunGraphOverlay(view);

  assert.equal(view.route, 'document');
  assert.equal(overlay.route, 'document');
  assert.deepEqual(view.nodes.map((node) => node.id), ['router', 'retrieval_worker']);
  assert.deepEqual(view.tools.map((tool) => tool.name), ['search_documents']);
  assert.equal(view.usedNodeCount, 2);
  assert.equal(view.usedToolCount, 1);
});
