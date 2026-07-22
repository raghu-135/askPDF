import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyTraceFocusToGraph,
  buildAgentGraph,
  getAgentGraphSpec,
} from '../src/components/agent-graph/agent-graph-mapper.ts';

const graphSpecs = {
  router: {
    nodes: [
      { id: 'context_loader', type: 'context_loader' },
      { id: 'router', type: 'router' },
      { id: 'retrieval_worker', type: 'retrieval_worker' },
      { id: 'memory_worker', type: 'memory_worker' },
      { id: 'timeline_worker', type: 'timeline_worker' },
      { id: 'web_worker', type: 'web_worker' },
      { id: 'direct_answer', type: 'direct_answer' },
      { id: 'synthesizer', type: 'synthesizer' },
      { id: 'finalizer', type: 'finalizer' },
    ],
    edges: [
      { from: 'START', to: 'context_loader' },
      { from: 'context_loader', to: 'router' },
      {
        from: 'router',
        conditional: true,
        routes: {
          document: 'retrieval_worker',
          memory: 'memory_worker',
          timeline: 'timeline_worker',
          web: 'web_worker',
          direct: 'direct_answer',
          clarify: 'finalizer',
        },
      },
      { from: 'retrieval_worker', to: 'synthesizer' },
      { from: 'memory_worker', to: 'synthesizer' },
      { from: 'timeline_worker', to: 'synthesizer' },
      { from: 'web_worker', to: 'synthesizer' },
      { from: 'direct_answer', to: 'finalizer' },
      { from: 'synthesizer', to: 'finalizer' },
      { from: 'finalizer', to: 'END' },
    ],
  },
  planExecute: {
    nodes: [
      { id: 'context_loader', type: 'context_loader' },
      { id: 'planner', type: 'planner' },
      { id: 'direct_answer', type: 'direct_answer' },
      { id: 'retrieval_worker', type: 'retrieval_worker' },
      { id: 'memory_worker', type: 'memory_worker' },
      { id: 'timeline_worker', type: 'timeline_worker' },
      { id: 'web_worker', type: 'web_worker' },
      { id: 'synthesizer', type: 'synthesizer' },
      { id: 'finalizer', type: 'finalizer' },
    ],
    edges: [
      { from: 'START', to: 'context_loader' },
      { from: 'context_loader', to: 'planner' },
      {
        from: 'planner',
        conditional: true,
        routes: { execute: 'retrieval_worker', direct: 'direct_answer', clarify: 'finalizer' },
      },
      { from: 'direct_answer', to: 'finalizer' },
      { from: 'retrieval_worker', to: 'memory_worker' },
      { from: 'memory_worker', to: 'timeline_worker' },
      { from: 'timeline_worker', to: 'web_worker' },
      { from: 'web_worker', to: 'synthesizer' },
      { from: 'synthesizer', to: 'finalizer' },
      { from: 'finalizer', to: 'END' },
    ],
  },
  evaluatorReplanner: {
    nodes: [
      { id: 'context_loader', type: 'context_loader' },
      { id: 'planner', type: 'planner' },
      { id: 'direct_answer', type: 'direct_answer' },
      { id: 'retrieval_worker', type: 'retrieval_worker' },
      { id: 'memory_worker', type: 'memory_worker' },
      { id: 'timeline_worker', type: 'timeline_worker' },
      { id: 'web_worker', type: 'web_worker' },
      { id: 'evidence_evaluator', type: 'evidence_evaluator' },
      { id: 'replanner', type: 'replanner' },
      { id: 'synthesizer', type: 'synthesizer' },
      { id: 'finalizer', type: 'finalizer' },
    ],
    edges: [
      { from: 'START', to: 'context_loader' },
      { from: 'context_loader', to: 'planner' },
      {
        from: 'planner',
        conditional: true,
        routes: { execute: 'retrieval_worker', direct: 'direct_answer', clarify: 'finalizer' },
      },
      { from: 'direct_answer', to: 'finalizer' },
      { from: 'retrieval_worker', to: 'memory_worker' },
      { from: 'memory_worker', to: 'timeline_worker' },
      { from: 'timeline_worker', to: 'web_worker' },
      { from: 'web_worker', to: 'evidence_evaluator' },
      {
        from: 'evidence_evaluator',
        conditional: true,
        routes: { answer: 'synthesizer', replan: 'replanner', answer_budget_exhausted: 'synthesizer' },
      },
      { from: 'replanner', to: 'retrieval_worker' },
      { from: 'synthesizer', to: 'finalizer' },
      { from: 'finalizer', to: 'END' },
    ],
  },
};

test('router graph maps conditional route edges and highlights selected route', () => {
  const graph = buildAgentGraph(
    graphSpecs.router,
    {
      route: 'document',
      nodeRows: [
        { node: 'context_loader', elapsed_ms: 2 },
        { node: 'router', elapsed_ms: 5, route: 'document' },
        { node: 'retrieval_worker', elapsed_ms: 9 },
        { node: 'synthesizer', elapsed_ms: 11 },
        { node: 'finalizer', elapsed_ms: 1 },
      ],
      toolRows: [
        { tool_name: 'search_documents', caller_node: 'retrieval_worker', ok: true, elapsed_ms: 8 },
      ],
    },
  );

  const selectedEdge = graph.edges.find((edge) => edge.route === 'document');
  const memoryEdge = graph.edges.find((edge) => edge.route === 'memory');
  const retrievalNode = graph.nodes.find((node) => node.id === 'retrieval_worker');

  assert.equal(selectedEdge?.selected, true);
  assert.equal(selectedEdge?.target, 'retrieval_worker');
  assert.equal(memoryEdge?.selected, false);
  assert.equal(retrievalNode?.status, 'completed');
  assert.equal(retrievalNode?.toolSummaries.length, 1);
});

test('plan execute graph marks planner plan and skipped workers', () => {
  const graph = buildAgentGraph(
    graphSpecs.planExecute,
    {
      route: 'execute',
      nodeRows: [
        { node: 'context_loader', elapsed_ms: 2 },
        { node: 'planner', elapsed_ms: 4, route: 'execute', execution_plan: ['retrieval_worker'] },
        { node: 'retrieval_worker', elapsed_ms: 8 },
        { node: 'memory_worker', elapsed_ms: 1, skipped: true, skip_reason: 'not_selected_by_plan' },
        { node: 'timeline_worker', elapsed_ms: 1, skipped: true, skip_reason: 'not_selected_by_plan' },
        { node: 'web_worker', elapsed_ms: 1, skipped: true, skip_reason: 'web_search_disabled' },
      ],
      toolRows: [],
    },
  );

  const planner = graph.nodes.find((node) => node.id === 'planner');
  const retrieval = graph.nodes.find((node) => node.id === 'retrieval_worker');
  const memory = graph.nodes.find((node) => node.id === 'memory_worker');
  const executeEdge = graph.edges.find((edge) => edge.route === 'execute');

  assert.deepEqual(graph.executionPlan, ['retrieval_worker']);
  assert.deepEqual(planner?.executionPlan, ['retrieval_worker']);
  assert.equal(retrieval?.status, 'completed');
  assert.equal(memory?.status, 'skipped');
  assert.equal(executeEdge?.selected, true);
});

test('evaluator replanner graph marks evaluator branch and replan plan', () => {
  const graph = buildAgentGraph(
    graphSpecs.evaluatorReplanner,
    {
      route: 'execute',
      nodeRows: [
        { node: 'planner', elapsed_ms: 4, route: 'execute', execution_plan: ['retrieval_worker'] },
        { node: 'retrieval_worker', elapsed_ms: 8 },
        { node: 'memory_worker', elapsed_ms: 1, skipped: true, skip_reason: 'not_selected_by_plan' },
        { node: 'timeline_worker', elapsed_ms: 1, skipped: true, skip_reason: 'not_selected_by_plan' },
        { node: 'web_worker', elapsed_ms: 1, skipped: true, skip_reason: 'web_search_disabled' },
        {
          node: 'evidence_evaluator',
          elapsed_ms: 5,
          evaluator_route: 'replan',
          evaluator_report: { sufficient: false, confidence: 0.3 },
        },
        { node: 'replanner', elapsed_ms: 4, execution_plan: ['timeline_worker'] },
      ],
      toolRows: [],
    },
  );

  const evaluator = graph.nodes.find((node) => node.id === 'evidence_evaluator');
  const replanner = graph.nodes.find((node) => node.id === 'replanner');
  const replanEdge = graph.edges.find((edge) => edge.source === 'evidence_evaluator' && edge.route === 'replan');

  assert.equal(evaluator?.label, 'Evidence Evaluator');
  assert.equal(evaluator?.status, 'completed');
  assert.equal(replanner?.status, 'completed');
  assert.deepEqual(replanner?.executionPlan, ['timeline_worker']);
  assert.equal(replanEdge?.selected, true);
});

test('graph mapper accepts trace-native graph rows', () => {
  const graph = buildAgentGraph(
    graphSpecs.router,
    {
      route: 'document',
      nodeRows: [
        {
          node: 'router',
          route: 'document',
          elapsed_ms: 4,
          llm_summary: {
            model_name: 'gpt-test',
            token_counts: { total: 42 },
            reasoning_available: true,
            retry_count: 1,
          },
        },
        { node: 'retrieval_worker', elapsed_ms: 9 },
      ],
      toolRows: [
        { tool_name: 'search_documents', caller_node: 'retrieval_worker', ok: true, source_count: 2 },
      ],
    },
  );

  const selectedEdge = graph.edges.find((edge) => edge.route === 'document');
  const retrievalNode = graph.nodes.find((node) => node.id === 'retrieval_worker');

  assert.equal(selectedEdge?.selected, true);
  assert.equal(retrievalNode?.status, 'completed');
  assert.equal(retrievalNode?.sourceCount, 2);
  assert.equal(retrievalNode?.toolSummaries.length, 1);
  assert.equal(graph.nodes.find((node) => node.id === 'router')?.llmSummary?.model_name, 'gpt-test');
  assert.equal(graph.nodes.find((node) => node.id === 'router')?.llmSummary?.token_counts.total, 42);
});

test('graph mapper labels custom node instances by catalog type while preserving instance ids', () => {
  const graph = buildAgentGraph(
    {
      nodes: [
        { id: 'context_1', type: 'context_loader' },
        { id: 'router_1', type: 'router' },
        { id: 'retrieval_1', type: 'retrieval_worker' },
        { id: 'final_1', type: 'finalizer' },
      ],
      edges: [
        { from: 'START', to: 'context_1' },
        { from: 'context_1', to: 'router_1' },
        {
          from: 'router_1',
          conditional: true,
          routes: { document: 'retrieval_1', direct: 'final_1' },
        },
        { from: 'retrieval_1', to: 'final_1' },
        { from: 'final_1', to: 'END' },
      ],
    },
    {
      nodeCatalog: {
        retrieval_worker: {
          display_name: 'Catalog Document Retrieval',
          category: 'retrieval',
          capabilities: ['retrieval.document'],
          observability: { span_kind: 'tool_worker', event_prefix: 'retrieval_worker' },
        },
        router: {
          display_name: 'Router',
          capabilities: 'malformed',
          observability: ['bad'],
        },
      },
      route: 'document',
      nodeRows: [
        { node: 'router_1', node_type: 'router', route: 'document', elapsed_ms: 4 },
        { node: 'retrieval_1', node_type: 'retrieval_worker', elapsed_ms: 9 },
      ],
      toolRows: [
        {
          tool_name: 'search_documents',
          caller_node: 'retrieval_1',
          caller_node_type: 'retrieval_worker',
          ok: true,
          source_count: 1,
        },
      ],
    },
  );

  const retrievalNode = graph.nodes.find((node) => node.id === 'retrieval_1');
  const selectedEdge = graph.edges.find((edge) => edge.route === 'document');

  assert.equal(retrievalNode?.label, 'Catalog Document Retrieval');
  assert.equal(retrievalNode?.category, 'retrieval');
  assert.deepEqual(retrievalNode?.capabilities, ['retrieval.document']);
  assert.equal(retrievalNode?.observability?.span_kind, 'tool_worker');
  assert.equal(retrievalNode?.instanceLabel, 'retrieval_1 · retrieval_worker');
  assert.equal(retrievalNode?.toolSummaries[0]?.callerNode, 'retrieval_1');
  assert.equal(retrievalNode?.toolSummaries[0]?.callerNodeType, 'retrieval_worker');
  assert.equal(graph.nodes.find((node) => node.id === 'router_1')?.capabilities, undefined);
  assert.equal(graph.nodes.find((node) => node.id === 'router_1')?.observability, undefined);
  assert.equal(selectedEdge?.selected, true);
  assert.equal(selectedEdge?.target, 'retrieval_1');
});

test('graph mapper applies node and span focus refs', () => {
  const graph = buildAgentGraph(
    graphSpecs.planExecute,
    {
      route: 'execute',
      nodeRows: [
        { node: 'planner', elapsed_ms: 4, __trace_span: { span_id: 'node:planner:0' } },
        { node: 'retrieval_worker', elapsed_ms: 8, __trace_span: { span_id: 'node:retrieval_worker:0' } },
      ],
      toolRows: [
        {
          tool_name: 'search_documents',
          caller_node: 'retrieval_worker',
          ok: true,
          __trace_span: { span_id: 'tool:search_documents:0', output: { sources: 2 } },
        },
      ],
    },
  );

  const focused = applyTraceFocusToGraph(graph, {
    node_ids: ['planner'],
    span_ids: ['tool:search_documents:0'],
  });

  const planner = focused.nodes.find((node) => node.id === 'planner');
  const retrieval = focused.nodes.find((node) => node.id === 'retrieval_worker');
  const memory = focused.nodes.find((node) => node.id === 'memory_worker');

  assert.equal(planner?.focused, true);
  assert.equal(retrieval?.focused, true);
  assert.deepEqual(retrieval?.focusedSpanIds, ['tool:search_documents:0']);
  assert.equal(retrieval?.focusedTraceSpans?.[0]?.output.sources, 2);
  assert.equal(memory?.focused, undefined);
});

test('graph mapper collapses repeated loop visits into one node with visit details', () => {
  const graph = buildAgentGraph(
    graphSpecs.evaluatorReplanner,
    {
      route: 'execute',
      nodeRows: [
        {
          node: 'evidence_evaluator',
          node_type: 'evidence_evaluator',
          visit_index: 1,
          elapsed_ms: 6,
          evaluator_route: 'replan',
          replan_count: 0,
          __trace_span: { span_id: 'node:evidence_evaluator:4' },
        },
        {
          node: 'replanner',
          node_type: 'replanner',
          visit_index: 1,
          elapsed_ms: 9,
          replan_count: 1,
          __trace_span: { span_id: 'node:replanner:5' },
        },
        {
          node: 'evidence_evaluator',
          node_type: 'evidence_evaluator',
          visit_index: 2,
          elapsed_ms: 7,
          evaluator_route: 'answer',
          replan_count: 1,
          warnings: ['low_confidence'],
          __trace_span: { span_id: 'node:evidence_evaluator:8' },
        },
      ],
      toolRows: [
        {
          tool_name: 'search_documents',
          caller_node: 'evidence_evaluator',
          caller_node_type: 'evidence_evaluator',
          caller_visit_index: 1,
          ok: true,
          elapsed_ms: 3,
        },
        {
          tool_name: 'search_thread_timeline',
          caller_node: 'evidence_evaluator',
          caller_node_type: 'evidence_evaluator',
          caller_visit_index: 2,
          ok: true,
          elapsed_ms: 4,
        },
      ],
    },
  );

  const evaluator = graph.nodes.find((node) => node.id === 'evidence_evaluator');

  assert.equal(evaluator?.visitCount, 2);
  assert.equal(evaluator?.latestVisitIndex, 2);
  assert.deepEqual(evaluator?.visits?.map((visit) => visit.visitIndex), [1, 2]);
  assert.deepEqual(evaluator?.visits?.map((visit) => visit.evaluatorRoute), ['replan', 'answer']);
  assert.deepEqual(evaluator?.visits?.map((visit) => visit.elapsedMs), [6, 7]);
  assert.deepEqual(evaluator?.visits?.map((visit) => visit.replanCount), [0, 1]);
  assert.deepEqual(evaluator?.visits?.map((visit) => visit.toolSummaries.map((tool) => tool.toolName)), [
    ['search_documents'],
    ['search_thread_timeline'],
  ]);
  assert.equal(evaluator?.visits?.[1]?.warningCount, 1);

  const focused = applyTraceFocusToGraph(graph, {
    span_ids: ['node:evidence_evaluator:8'],
  });
  const focusedEvaluator = focused.nodes.find((node) => node.id === 'evidence_evaluator');

  assert.equal(focusedEvaluator?.focused, true);
  assert.deepEqual(focusedEvaluator?.focusedSpanIds, ['node:evidence_evaluator:8']);
});

test('graph mapper keeps visit-indexed tools visible when node visit rows are missing', () => {
  const graph = buildAgentGraph(
    graphSpecs.evaluatorReplanner,
    {
      toolRows: [
        {
          tool_name: 'search_documents',
          caller_node: 'evidence_evaluator',
          caller_node_type: 'evidence_evaluator',
          caller_visit_index: 2,
          ok: true,
        },
      ],
    },
  );

  const evaluator = graph.nodes.find((node) => node.id === 'evidence_evaluator');

  assert.equal(evaluator?.visitCount, 1);
  assert.equal(evaluator?.visits?.[0]?.visitIndex, 2);
  assert.deepEqual(evaluator?.visits?.[0]?.toolSummaries.map((tool) => tool.toolName), ['search_documents']);
});

test('graph spec mapper does not synthesize legacy builtins without stored topology', () => {
  assert.deepEqual(getAgentGraphSpec({ workflow_type: 'router_rag_agent' }), { nodes: [], edges: [] });
  assert.deepEqual(
    getAgentGraphSpec({ config: { graph: graphSpecs.router } }),
    graphSpecs.router,
  );
});
