import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildTraceExportJson,
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
        llm_span_count: 1,
        llm_token_count_prompt: 100,
        llm_token_count_completion: 25,
        llm_token_count_total: 125,
        llm_token_count_reasoning: 9,
        llm_token_count_cached: 11,
        llm_retry_count: 1,
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
            {
              name: 'llm.retry',
              attributes: {
                'llm.retry.attempt': 1,
                'llm.retry.delay_ms': 2000,
                'llm.retry.reason': 'Retryable OpenAI-compatible API error (429)',
                'http.status_code': 429,
                'exception.type': 'RuntimeError',
              },
            },
            {
              name: 'llm.completed',
              attributes: {
                'llm.model_name': 'gpt-test',
                'llm.response_chars': 42,
                'llm.token_count.prompt': 100,
                'llm.token_count.completion': 25,
                'llm.token_count.total': 125,
                'llm.token_count.reasoning': 9,
                'llm.token_count.cached': 11,
                'llm.reasoning_available': true,
                'llm.reasoning_format': 'structured',
                'llm.reasoning_chars': 300,
              },
              output: {
                reasoning_preview: 'I inspected the retrieved evidence and selected the document path.',
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
    },
  },
};

test('trace projection builds run view from debug.trace spans', () => {
  const view = buildRunTraceView(traceBackedRun);

  assert.equal(view.route, 'execute');
  assert.equal(view.routeReason, 'Document evidence requested.');
  assert.deepEqual(view.nodes.map((node) => node.id), ['planner', 'retrieval_worker', 'memory_worker']);
  assert.deepEqual(view.tools.map((tool) => tool.name), ['search_documents']);
  assert.equal(view.nodes[0].span?.span_id, 'node:planner:0');
  assert.equal(view.tools[0].span?.span_id, 'tool:search_documents:0');
});

test('trace projection exposes llm usage and retry metadata on node rows', () => {
  const view = buildRunTraceView(traceBackedRun);
  const planner = view.nodes.find((node) => node.id === 'planner');

  assert.equal(planner?.raw.llm_summary.model_name, 'gpt-test');
  assert.equal(planner?.raw.llm_summary.response_chars, 42);
  assert.equal(planner?.raw.llm_summary.token_counts.prompt, 100);
  assert.equal(planner?.raw.llm_summary.token_counts.completion, 25);
  assert.equal(planner?.raw.llm_summary.token_counts.total, 125);
  assert.equal(planner?.raw.llm_summary.token_counts.reasoning, 9);
  assert.equal(planner?.raw.llm_summary.token_counts.cached, 11);
  assert.equal(planner?.raw.llm_summary.reasoning_available, true);
  assert.equal(planner?.raw.llm_summary.reasoning_format, 'structured');
  assert.equal(planner?.raw.llm_summary.reasoning_chars, 300);
  assert.equal(planner?.raw.llm_summary.reasoning_preview, 'I inspected the retrieved evidence and selected the document path.');
  assert.equal(planner?.raw.llm_summary.retry_attempts.length, 1);
  assert.equal(planner?.raw.llm_summary.retry_attempts[0].http_status_code, 429);
});

test('trace summary counts non-skipped nodes and available graph nodes', () => {
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

test('trace export returns only normalized trace json', () => {
  const view = buildRunTraceView(traceBackedRun);
  const exported = JSON.parse(buildTraceExportJson(view));

  assert.equal(exported.trace_id, 'run-1');
  assert.equal(exported.schema_version, 1);
  assert.equal(exported.spans.length, traceBackedRun.debug.trace.spans.length);
  assert.equal(exported.node_events, undefined);
  assert.equal(exported.tool_events, undefined);
});
