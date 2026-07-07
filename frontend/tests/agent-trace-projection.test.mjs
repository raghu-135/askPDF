import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildTraceExportJson,
  buildRunTraceView,
} from '../src/components/agent-debug/agent-trace-projection.ts';

const backendDebug = {
  version: 1,
  trace: {
    schema_version: 1,
    trace_id: 'run-1',
    run_id: 'run-1',
    status: 'completed',
    metrics: {
      duration_ms: 25,
      tool_event_count: 1,
      tool_warning_count: 0,
      error_count: 0,
      llm_span_count: 1,
      llm_token_count_total: 125,
      llm_retry_count: 1,
    },
    spans: [
      { span_id: 'run:run-1', name: 'Agent Run', kind: 'AGENT', status: 'completed' },
      { span_id: 'node:planner:0', name: 'Planner', kind: 'AGENT', status: 'completed' },
      { span_id: 'tool:search_documents:0', name: 'Document Evidence', kind: 'TOOL', status: 'completed' },
    ],
  },
  summary: {
    status: 'completed',
    route: 'execute',
    routeReason: 'Document evidence requested.',
    metrics: {
      duration_ms: 25,
      tool_event_count: 1,
      tool_warning_count: 0,
      error_count: 0,
      llm_span_count: 1,
      llm_token_count_total: 125,
      llm_retry_count: 1,
    },
    nodes: [
      {
        id: 'planner',
        status: 'completed',
        skipped: false,
        durationMs: 5,
        route: 'execute',
        routeReason: 'Document evidence requested.',
        executionPlan: ['retrieval_worker'],
        warningCodes: [],
        span: { span_id: 'node:planner:0' },
        raw: {
          node: 'planner',
          route: 'execute',
          route_reason: 'Document evidence requested.',
          execution_plan: ['retrieval_worker'],
          llm_result_summary: {
            llm: {
              model_name: 'gpt-test',
              token_counts: { prompt: 100, completion: 25, total: 125 },
              retry_count: 1,
            },
          },
        },
      },
      {
        id: 'retrieval_worker',
        status: 'completed',
        skipped: false,
        durationMs: 8,
        warningCodes: [],
        span: { span_id: 'node:retrieval_worker:1' },
        raw: { node: 'retrieval_worker', output_preview: { evidence: 'Found document evidence.' } },
      },
      {
        id: 'memory_worker',
        status: 'skipped',
        skipped: true,
        durationMs: 0.5,
        warningCodes: [],
        span: { span_id: 'node:memory_worker:2' },
        raw: { node: 'memory_worker', skip_reason: 'not_selected_by_plan' },
      },
    ],
    tools: [
      {
        name: 'search_documents',
        id: 'document_evidence',
        category: 'document',
        displayName: 'Document Evidence',
        callerNode: 'retrieval_worker',
        ok: true,
        durationMs: 7,
        sourceCount: 1,
        warningCodes: [],
        span: { span_id: 'tool:search_documents:0' },
        raw: {
          tool_name: 'search_documents',
          artifact_keys: ['document_sources'],
          result_preview: 'Found document evidence.',
        },
      },
    ],
    usedNodeCount: 2,
    availableNodeCount: 4,
    usedToolCount: 1,
    availableToolCount: 2,
    warningCount: 0,
    errorCount: 0,
    errors: [],
  },
  graph: {
    nodes: [
      {
        id: 'planner',
        type: 'planner',
        label: 'Planner',
        status: 'active',
        toolSummaries: [],
        warningCount: 0,
        errorCount: 0,
        sourceCount: 0,
        artifactCount: 0,
        rawEvents: [],
      },
      {
        id: 'retrieval_worker',
        type: 'retrieval_worker',
        label: 'Document Retrieval',
        status: 'active',
        toolSummaries: [
          {
            toolName: 'search_documents',
            ok: true,
            warnings: [],
            artifactKeys: ['document_sources'],
            raw: {},
          },
        ],
        warningCount: 0,
        errorCount: 0,
        sourceCount: 1,
        artifactCount: 1,
        rawEvents: [],
      },
    ],
    edges: [
      {
        id: 'planner-execute-retrieval_worker',
        source: 'planner',
        target: 'retrieval_worker',
        label: 'execute',
        route: 'execute',
        selected: true,
        active: true,
        conditional: true,
      },
    ],
    executionPlan: ['retrieval_worker'],
    selectedRoute: 'execute',
  },
};

const traceBackedRun = {
  id: 'run-1',
  workflow_id: 'plan_execute_rag_agent',
  metrics_json: { duration_ms: 99 },
  debug: backendDebug,
};

test('trace projection reads backend-provided summary and graph', () => {
  const view = buildRunTraceView(traceBackedRun);

  assert.equal(view.route, 'execute');
  assert.equal(view.routeReason, 'Document evidence requested.');
  assert.deepEqual(view.nodes.map((node) => node.id), ['planner', 'retrieval_worker', 'memory_worker']);
  assert.deepEqual(view.tools.map((tool) => tool.name), ['search_documents']);
  assert.equal(view.nodes[0].span?.span_id, 'node:planner:0');
  assert.equal(view.tools[0].span?.span_id, 'tool:search_documents:0');
  assert.equal(view.graph?.selectedRoute, 'execute');
  assert.equal(view.graph?.nodes[1].toolSummaries.length, 1);
});

test('trace projection preserves custom node type metadata and normalizes graph labels', () => {
  const customDebug = {
    ...backendDebug,
    summary: {
      ...backendDebug.summary,
      nodes: [
        {
          id: 'retrieval_1',
          type: 'retrieval_worker',
          status: 'completed',
          skipped: false,
          durationMs: 8,
          warningCodes: [],
          span: { span_id: 'node:retrieval_1:0' },
          raw: {
            node: 'retrieval_1',
            node_type: 'retrieval_worker',
            output_preview: { evidence: 'Found custom document evidence.' },
          },
        },
      ],
      tools: [
        {
          name: 'search_documents',
          id: 'document_evidence',
          displayName: 'Document Evidence',
          callerNode: 'retrieval_1',
          callerNodeType: 'retrieval_worker',
          ok: true,
          durationMs: 7,
          sourceCount: 1,
          warningCodes: [],
          span: { span_id: 'tool:search_documents:0' },
          raw: {
            tool_name: 'search_documents',
            caller_node: 'retrieval_1',
            caller_node_type: 'retrieval_worker',
            artifact_keys: ['document_sources'],
          },
        },
      ],
    },
    graph: {
      ...backendDebug.graph,
      nodes: [
        {
          id: 'retrieval_1',
          type: 'retrieval_worker',
          label: 'Retrieval 1',
          capabilities: ['retrieval.document'],
          observability: { span_kind: 'tool_worker' },
          status: 'active',
          toolSummaries: [],
          warningCount: 0,
          errorCount: 0,
          sourceCount: 1,
          artifactCount: 1,
          rawEvents: [],
        },
      ],
      edges: [],
    },
  };

  const view = buildRunTraceView(
    { ...traceBackedRun, id: 'run-custom', debug: customDebug },
    {
      nodeCatalog: {
        retrieval_worker: {
          display_name: 'Catalog Document Retrieval',
          category: 'retrieval',
          capabilities: ['catalog.capability'],
          observability: { event_prefix: 'retrieval_worker' },
        },
      },
    },
  );

  assert.equal(view.nodes[0].id, 'retrieval_1');
  assert.equal(view.nodes[0].type, 'retrieval_worker');
  assert.equal(view.nodes[0].label, 'Catalog Document Retrieval');
  assert.equal(view.nodes[0].instanceLabel, 'retrieval_1 · retrieval_worker');
  assert.equal(view.tools[0].callerNode, 'retrieval_1');
  assert.equal(view.tools[0].callerNodeType, 'retrieval_worker');
  assert.equal(view.graph?.nodes[0].label, 'Catalog Document Retrieval');
  assert.equal(view.graph?.nodes[0].category, 'retrieval');
  assert.deepEqual(view.graph?.nodes[0].capabilities, ['retrieval.document']);
  assert.equal(view.graph?.nodes[0].observability?.span_kind, 'tool_worker');
  assert.equal(view.graph?.nodes[0].instanceLabel, 'retrieval_1 · retrieval_worker');
});

test('trace projection uses backend counts without inferring from spans', () => {
  const view = buildRunTraceView(traceBackedRun);

  assert.equal(view.usedNodeCount, 2);
  assert.equal(view.availableNodeCount, 4);
  assert.equal(view.usedToolCount, 1);
  assert.equal(view.availableToolCount, 2);
  assert.equal(view.warningCount, 0);
  assert.equal(view.errorCount, 0);
  assert.equal(view.metrics.llm_token_count_total, 125);
  assert.equal(view.metrics.llm_retry_count, 1);
});

test('trace projection handles null debug payload', () => {
  assert.equal(buildRunTraceView({ id: 'run-empty', debug: null }), undefined);
});

test('trace projection rejects empty or stale debug payloads', () => {
  assert.equal(buildRunTraceView({ id: 'run-empty-object', debug: {} }), undefined);
  assert.equal(buildRunTraceView({ id: 'run-stale', debug: { ...backendDebug, version: 0 } }), undefined);
  assert.equal(buildRunTraceView({ id: 'run-partial', debug: { version: 1, trace: backendDebug.trace } }), undefined);
});

test('trace export returns full backend debug json', () => {
  const view = buildRunTraceView(traceBackedRun);
  const exported = JSON.parse(buildTraceExportJson(view));

  assert.equal(exported.version, 1);
  assert.equal(exported.trace.trace_id, 'run-1');
  assert.equal(exported.summary.route, 'execute');
  assert.equal(exported.graph.selectedRoute, 'execute');
  assert.equal(exported.node_events, undefined);
  assert.equal(exported.tool_events, undefined);
});
