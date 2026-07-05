import assert from 'node:assert/strict';
import test from 'node:test';

import { buildAgentGraph, getAgentGraphSpec } from '../src/components/agent-graph/agent-graph-mapper.ts';

test('router graph maps conditional route edges and highlights selected route', () => {
  const graph = buildAgentGraph(
    getAgentGraphSpec({ pattern_type: 'router_rag_agent' }),
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
  assert.equal(retrievalNode?.status, 'active');
  assert.equal(retrievalNode?.toolSummaries.length, 1);
});

test('plan execute graph marks planner plan and skipped workers', () => {
  const graph = buildAgentGraph(
    getAgentGraphSpec({ pattern_type: 'plan_execute_rag_agent' }),
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
  assert.equal(retrieval?.status, 'active');
  assert.equal(memory?.status, 'skipped');
  assert.equal(executeEdge?.selected, true);
});

test('graph mapper accepts trace-native graph rows', () => {
  const graph = buildAgentGraph(
    getAgentGraphSpec({ pattern_type: 'router_rag_agent' }),
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
  assert.equal(retrievalNode?.status, 'active');
  assert.equal(retrievalNode?.sourceCount, 2);
  assert.equal(retrievalNode?.toolSummaries.length, 1);
  assert.equal(graph.nodes.find((node) => node.id === 'router')?.llmSummary?.model_name, 'gpt-test');
  assert.equal(graph.nodes.find((node) => node.id === 'router')?.llmSummary?.token_counts.total, 42);
});
