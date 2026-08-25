import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildTraceExportJson,
  buildCorrectiveInspection,
  buildLiveTraceView,
  buildRunTraceView,
  getRetainedRunErrorMessage,
  mergeLiveAndRetainedTraceViews,
  shouldRefreshRetainedTrace,
} from '../src/components/agent-debug/agent-trace-projection.ts';

test('retained run errors remain visible when no trace was captured', () => {
  assert.equal(
    getRetainedRunErrorMessage({ error_json: { safe_message: 'Hermes timed out' } }),
    'Hermes timed out',
  );
  assert.equal(getRetainedRunErrorMessage({ error_json: {} }), null);
  assert.equal(shouldRefreshRetainedTrace({ id: 'run-1', status: 'failed' }), true);
  assert.equal(shouldRefreshRetainedTrace({ id: 'run-1', status: 'running' }), false);
});

test('corrective inspection preserves wave outcomes, packet grades, and exact claim sources', () => {
  const inspection = buildCorrectiveInspection({
    id: 'run-corrective',
    metrics_json: {
      corrective: {
        partial_waves: 1,
        source_expansions: 1,
        successful_waves: 1,
        wave_outcomes: [{ wave_id: 1, outcome: 'successful', latency_ms: 42, work_items: [{ query_id: 'query-1', source_strategy: 'web', status: 'completed' }] }],
      },
      retrieval_quality_report: {
        packet_assessments: [{ packet_id: 'packet-1', eligible: false, rejection_reasons: ['instruction_injection_risk'] }],
      },
      grounding_report: {
        claims: [{ claim_id: 'c1', claim: 'Supported', support: 'full', source_ids: ['doc:file:1'] }],
        contradictions: [{ claim: 'Conflict', claim_ids: ['c1'], source_ids: ['doc:file:1', 'web:https://example.com/'] }],
      },
    },
  });
  assert.equal(inspection.corrective.wave_outcomes[0].work_items[0].query_id, 'query-1');
  assert.equal(inspection.corrective.wave_outcomes[0].outcome, 'successful');
  assert.equal(inspection.corrective.wave_outcomes[0].latency_ms, 42);
  assert.deepEqual(inspection.retrievalQuality.packet_assessments[0].rejection_reasons, ['instruction_injection_risk']);
  assert.deepEqual(inspection.grounding.claims[0].source_ids, ['doc:file:1']);
  assert.equal(inspection.grounding.claims[0].claim_id, 'c1');
  assert.equal(inspection.grounding.contradictions.length, 1);
});

test('Hermes grounding inspection reads runtime evidence metrics', () => {
  const inspection = buildCorrectiveInspection({
    id: 'run-hermes',
    metrics_json: {
      grounding: {
        grounded: false,
        requirement: 'document',
        evidence_result_count: 0,
        failure_codes: ['tool_arguments_invalid'],
      },
    },
  });
  assert.equal(inspection.grounding.grounded, false);
  assert.equal(inspection.grounding.requirement, 'document');
  assert.deepEqual(inspection.grounding.failure_codes, ['tool_arguments_invalid']);
});

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
    operations: [
      {
        id: 'planner',
        status: 'completed',
        skipped: false,
        visitIndex: 1,
        durationMs: 5,
        route: 'execute',
        routeReason: 'Document evidence requested.',
        executionPlan: ['retrieval_worker'],
        warningCodes: [],
        span: { span_id: 'node:planner:0' },
        raw: {
          node: 'planner',
          visit_index: 1,
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
        id: 'thread_conversation_history_worker',
        status: 'skipped',
        skipped: true,
        durationMs: 0.5,
        warningCodes: [],
        span: { span_id: 'node:thread_conversation_history_worker:2' },
        raw: { node: 'thread_conversation_history_worker', skip_reason: 'not_selected_by_plan' },
      },
    ],
    tools: [
      {
        name: 'search_documents',
        id: 'document_evidence',
        category: 'document',
        displayName: 'Document Evidence',
        callerNode: 'retrieval_worker',
        callerVisitIndex: 1,
        ok: true,
        durationMs: 7,
        sourceCount: 1,
        warningCodes: [],
        span: { span_id: 'tool:search_documents:0' },
        raw: {
          tool_name: 'search_documents',
          caller_visit_index: 1,
          artifact_keys: ['document_sources'],
          result_preview: 'Found document evidence.',
        },
      },
    ],
    usedOperationCount: 2,
    availableOperationCount: 4,
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

const canonicalDebug = (source) => {
  const debug = structuredClone(source);
  const operations = (debug.summary?.operations || []).map((row) => ({
    operation_id: row.operation_id || row.id,
    operation_type: row.operation_type || row.type,
    operation_label: row.operation_label || row.label || row.id,
    visit_index: row.visit_index || row.visitIndex,
    status: row.status,
    duration_ms: row.duration_ms || row.durationMs,
    topology_ref: { kind: 'graph_node', id: row.operation_id || row.id },
    ...row,
  }));
  const graph = debug.graph;
  debug.version = 2;
  debug.events = [];
  debug.operations = operations;
  debug.summary = { ...debug.summary, operations };
  debug.visualizations = {
    'generic.timeline': { id: 'generic.timeline' },
    ...(graph ? { 'langgraph.graph': {
      id: 'langgraph.graph', nodes: graph.nodes, edges: graph.edges,
      execution_plan: graph.executionPlan, selected_route: graph.selectedRoute,
    } } : {}),
  };
  delete debug.graph;
  return debug;
};

const traceBackedRun = {
  id: 'run-1',
  workflow_id: 'plan_execute_rag_agent',
  metrics_json: { duration_ms: 99 },
  debug: canonicalDebug(backendDebug),
};

test('trace projection reads backend-provided summary and graph', () => {
  const view = buildRunTraceView(traceBackedRun);

  assert.equal(view.route, 'execute');
  assert.equal(view.routeReason, 'Document evidence requested.');
  assert.deepEqual(view.operations.map((operation) => operation.id), ['planner', 'retrieval_worker', 'thread_conversation_history_worker']);
  assert.deepEqual(view.tools.map((tool) => tool.name), ['search_documents']);
  assert.equal(view.operations[0].visitIndex, 1);
  assert.equal(view.tools[0].callerVisitIndex, 1);
  assert.equal(view.operations[0].span?.span_id, 'node:planner:0');
  assert.equal(view.tools[0].span?.span_id, 'tool:search_documents:0');
  assert.equal(view.graph?.selectedRoute, 'execute');
  assert.equal(view.graph?.nodes[1].toolSummaries.length, 1);
});

test('retained trace prefers correlated canonical failures over aggregate counters', () => {
  const failures = [
    { event_id: 'failure-1', sequence: 10, kind: 'tool.failed', classification: 'primary', error: { code: 'search_failed', message: 'Search unavailable' } },
    { event_id: 'failure-2', sequence: 11, kind: 'run.failed', classification: 'terminal', failure_count: 2, primary_failure_event_id: 'failure-1', contributing_failure_event_ids: ['failure-1'], error: { code: 'evidence_unavailable', message: 'Evidence unavailable' } },
  ];
  const debug = canonicalDebug({ ...backendDebug, failures, summary: { ...backendDebug.summary, errorCount: 99, errors: [] } });

  const view = buildRunTraceView({ ...traceBackedRun, debug });

  assert.equal(view.errorCount, 2);
  assert.deepEqual(view.errors.map((failure) => failure.event_id), ['failure-1', 'failure-2']);
  assert.deepEqual(view.errors[1].contributing_failure_event_ids, ['failure-1']);
});

test('trace projection preserves custom node type metadata and normalizes graph labels', () => {
  const customDebug = {
    ...backendDebug,
    summary: {
      ...backendDebug.summary,
      operations: [
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
    { ...traceBackedRun, id: 'run-custom', debug: canonicalDebug(customDebug) },
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

  assert.equal(view.operations[0].id, 'retrieval_1');
  assert.equal(view.operations[0].type, 'retrieval_worker');
  assert.equal(view.operations[0].label, 'retrieval_1');
  assert.equal(view.operations[0].instanceLabel, 'retrieval_1 · retrieval_worker');
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

  assert.equal(view.usedOperationCount, 2);
  assert.equal(view.availableOperationCount, 4);
  assert.equal(view.usedToolCount, 1);
  assert.equal(view.availableToolCount, 2);
  assert.equal(view.warningCount, 0);
  assert.equal(view.errorCount, 0);
  assert.equal(view.metrics.llm_token_count_total, 125);
  assert.equal(view.metrics.llm_retry_count, 1);
});

test('trace projection creates expandable rows for retained detail visits missing spans', () => {
  const view = buildRunTraceView({
    ...traceBackedRun,
    debug: {
      ...traceBackedRun.debug,
      detail_manifest: [
        { operation_id: 'planner', operation_type: 'deep_task_planner', visit_index: 1, status: 'completed', available: true },
        { operation_id: 'planner', operation_type: 'deep_task_planner', visit_index: 2, status: 'completed', available: true },
      ],
      operations: [],
      summary: { ...traceBackedRun.debug.summary, operations: [] },
    },
  });

  assert.deepEqual(view.operations.map((operation) => [operation.id, operation.visitIndex]), [['planner', 1], ['planner', 2]]);
  assert.equal(view.usedOperationCount, 2);
});

test('retained terminal node spans cannot remain visually active', () => {
  const completedRun = {
    ...backendDebug,
    summary: {
      ...backendDebug.summary,
      operations: [
        {
          id: 'serial_dispatch',
          type: 'serial_dispatch',
          status: 'running',
          skipped: false,
          warningCodes: [],
          span: { end_time: '2026-08-06T20:00:01Z' },
          raw: { dispatch_status: 'running', visit_index: 1 },
        },
        {
          id: 'serial_dispatch',
          type: 'serial_dispatch',
          status: 'ready_to_aggregate',
          skipped: false,
          warningCodes: [],
          span: { end_time: '2026-08-06T20:00:02Z' },
          raw: { dispatch_status: 'ready_to_aggregate', visit_index: 2 },
        },
      ],
    },
  };

  const view = buildRunTraceView({ ...traceBackedRun, debug: canonicalDebug(completedRun) });

  assert.deepEqual(view.operations.map((operation) => operation.status), ['completed', 'completed']);
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

  assert.equal(exported.version, 2);
  assert.equal(exported.trace.trace_id, 'run-1');
  assert.equal(exported.summary.route, 'execute');
  assert.equal(exported.visualizations['langgraph.graph'].selected_route, 'execute');
  assert.equal(exported.node_events, undefined);
  assert.equal(exported.tool_events, undefined);
});

test('live trace projection keeps loop visits, full details, tools, and final output', () => {
  const detail = (visit) => ({
    node_id: 'evidence_evaluator',
    node_type: 'evidence_evaluator',
    visit_index: visit,
    status: 'completed',
    checkpoint_before: { replan_count: visit - 1 },
    checkpoint_after: { replan_count: visit },
  });
  const view = buildLiveTraceView([
    { id: 1, event: 'operation.started', data: { operation_id: 'evidence_evaluator', operation_type: 'evidence_evaluator', visit_index: 1 } },
    { id: 2, event: 'operation.completed', data: { operation_id: 'evidence_evaluator', operation_type: 'evidence_evaluator', visit_index: 1, evaluator_route: 'replan', detail: detail(1) } },
    { id: 3, event: 'tool.completed', data: { tool_name: 'search_documents', caller_node: 'evidence_evaluator', caller_visit_index: 1, ok: true } },
    { id: 4, event: 'operation.started', data: { operation_id: 'evidence_evaluator', operation_type: 'evidence_evaluator', visit_index: 2 } },
    { id: 5, event: 'operation.completed', data: { operation_id: 'evidence_evaluator', operation_type: 'evidence_evaluator', visit_index: 2, evaluator_route: 'answer', detail: detail(2) } },
    { id: 6, event: 'run.completed', data: { final_output: { answer: 'Complete final answer', route: 'document' } } },
  ]);

  assert.deepEqual(view.operations.map((operation) => operation.visitIndex), [1, 2]);
  assert.equal(view.operations[0].raw.detail.checkpoint_after.replan_count, 1);
  assert.equal(view.operations[1].raw.detail.checkpoint_after.replan_count, 2);
  assert.equal(view.tools[0].callerVisitIndex, 1);
  assert.equal(view.finalOutput.answer, 'Complete final answer');
  assert.deepEqual(view.detailManifest.map((row) => row.visit_index), [1, 2]);
});

test('live trace projection preserves string node failures', () => {
  const view = buildLiveTraceView([
    { id: 1, event: 'operation.failed', data: { operation_id: 'router', operation_type: 'router', visit_index: 1, error: 'Model connection failed' } },
  ]);

  assert.equal(view.operations[0].status, 'error');
  assert.equal(view.operations[0].error.raw_message, 'Model connection failed');
  assert.equal(view.errors[0].raw_message, 'Model connection failed');
});

test('live trace projection surfaces terminal run failures without a node failure', () => {
  const view = buildLiveTraceView([
    { id: 1, event: 'operation.completed', data: { operation_id: 'router', operation_type: 'router', visit_index: 1, route: 'memory' } },
    { id: 2, event: 'run.failed', data: { error: { code: 'workflow_failed', raw_message: 'No destination for route memory' } } },
  ]);

  assert.equal(view.operations[0].status, 'completed');
  assert.equal(view.errorCount, 1);
  assert.equal(view.errors[0].code, 'workflow_failed');
  assert.equal(view.errors[0].raw_message, 'No destination for route memory');
});

test('live and retained projections merge repeated visits without duplicating identities', () => {
  const retained = buildLiveTraceView([
    { id: 1, event: 'operation.completed', data: { operation_id: 'router', operation_type: 'router', visit_index: 1, route: 'execute' } },
    { id: 2, event: 'operation.completed', data: { operation_id: 'worker', operation_type: 'document_retrieval', visit_index: 1 } },
  ]);
  const live = buildLiveTraceView([
    { id: 3, event: 'operation.started', data: { operation_id: 'worker', operation_type: 'document_retrieval', visit_index: 1 } },
    { id: 4, event: 'operation.completed', data: { operation_id: 'worker', operation_type: 'document_retrieval', visit_index: 1 } },
    { id: 5, event: 'operation.completed', data: { operation_id: 'worker', operation_type: 'document_retrieval', visit_index: 2 } },
  ]);

  const merged = mergeLiveAndRetainedTraceViews(live, retained);

  assert.deepEqual(merged.operations.map((operation) => `${operation.id}:${operation.visitIndex}`), ['router:1', 'worker:1', 'worker:2']);
  assert.equal(merged.operations[1].status, 'completed');
});

test('runtime-neutral operation events project without graph semantics', () => {
  const view = buildLiveTraceView([
    { id: 1, event: 'operation.started', data: { operation_id: 'hermes_session', operation_type: 'agent_session', operation_label: 'Hermes Agent', visit_index: 1 } },
    { id: 2, event: 'operation.completed', data: { operation_id: 'hermes_session', operation_type: 'agent_session', operation_label: 'Hermes Agent', visit_index: 1 } },
  ]);

  assert.equal(view.operations.length, 1);
  assert.equal(view.operations[0].id, 'hermes_session');
  assert.equal(view.operations[0].label, 'Hermes Agent');
  assert.equal(view.operations[0].status, 'completed');
  assert.equal(view.graph, undefined);
});

test('live trace projection correlates parallel worker progress by work id', () => {
  const view = buildLiveTraceView([
    { id: 1, event: 'dispatch.started', data: { dispatch_id: 'dispatch-1', planned: 2 } },
    { id: 2, event: 'worker.queued', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents' } },
    { id: 3, event: 'worker.started', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents', attempt: 1 } },
    { id: 4, event: 'worker.completed', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents', attempt: 1, elapsed_ms: 12 } },
    { id: 5, event: 'aggregation.partial', data: { dispatch_id: 'dispatch-1', planned: 2, completed: 1, failed: 1, partial_evidence: true } },
  ]);

  assert.equal(view.parallel.summary.dispatch_id, 'dispatch-1');
  assert.equal(view.parallel.summary.partial_evidence, true);
  assert.equal(view.parallel.tasks.length, 1);
  assert.equal(view.parallel.tasks[0].work_id, 'work-1');
  assert.equal(view.parallel.tasks[0].status, 'completed');
  assert.equal(view.parallel.tasks[0].elapsed_ms, 12);
});

test('retained trace projection restores expandable parallel attempts', () => {
  const debug = canonicalDebug(backendDebug);
  debug.summary.metrics = {
    parallel_summary: { dispatch_id: 'dispatch-1', planned: 1, completed: 1, retried: 1 },
    parallel_attempts: [
      { event: 'worker.started', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents', attempt: 1 } },
      { event: 'worker.retrying', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents', attempt: 1 } },
      { event: 'worker.started', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents', attempt: 2 } },
      { event: 'worker.completed', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, worker_node_id: 'documents', attempt: 2, elapsed_ms: 18 } },
      { event: 'dispatch.barrier_reached', data: { dispatch_id: 'dispatch-1', result_count: 1 } },
    ],
  };
  const view = buildRunTraceView({ id: 'run-1', thread_id: 'thread-1', workflow_id: 'workflow-1', status: 'completed', debug });

  assert.equal(view.parallel.summary.event, 'dispatch.barrier_reached');
  assert.equal(view.parallel.tasks[0].attempts.length, 2);
  assert.equal(view.parallel.tasks[0].attempts[0].status, 'retrying');
  assert.equal(view.parallel.tasks[0].attempts[1].status, 'completed');
});

test('parallel projection preserves terminal attempts and lifecycle under out-of-order events', () => {
  const view = buildLiveTraceView([
    { id: 1, event: 'worker.timed_out', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, attempt: 1 } },
    { id: 2, event: 'aggregation.partial', data: { dispatch_id: 'dispatch-1', planned: 1, timed_out: 1, partial_evidence: true } },
    { id: 3, event: 'worker.started', data: { dispatch_id: 'dispatch-1', work_id: 'work-1', ordinal: 0, attempt: 1 } },
    { id: 4, event: 'dispatch.started', data: { dispatch_id: 'dispatch-1', planned: 1 } },
  ]);

  assert.equal(view.parallel.tasks[0].status, 'timed_out');
  assert.equal(view.parallel.tasks[0].attempts[0].status, 'timed_out');
  assert.equal(view.parallel.summary.barrier_state, 'reached');
  assert.equal(view.parallel.summary.aggregation_state, 'partial');
});
