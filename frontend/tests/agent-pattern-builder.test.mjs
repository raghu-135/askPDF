import assert from 'node:assert/strict';
import test from 'node:test';

import {
  assembleAgentPatternSpec,
  canAddNodeType,
  canConnectNodes,
  createHitlGateForTarget,
  createInitialBuilderState,
  getAllowedRouteFunctionsForNode,
  getAllowedToolContractsForNode,
  getCanonicalNodeId,
  getRouteLabelsForFunction,
  loadBuilderStateFromSpec,
  normalizeBuilderState,
} from '../src/lib/agent-pattern-builder.ts';

const node = (overrides) => ({
  display_name: overrides.display_name || overrides.type,
  category: overrides.category || 'test',
  capabilities: overrides.capabilities || [],
  allowed_route_functions: overrides.allowed_route_functions || [],
  allowed_tool_contract_ids: overrides.allowed_tool_contract_ids || [],
  allowed_parent_types: overrides.allowed_parent_types || [],
  allowed_child_types: overrides.allowed_child_types || [],
  limits: { default_max_visits: 1 },
  state_reads: [],
  state_writes: [],
  prompt_slots: [],
  context_policy: { mode: 'test', input_budget: 'test', output_budget: 'test' },
  observability: { span_kind: 'test', event_prefix: overrides.type, summary_fields: [], raw_payload: 'bounded' },
  max_instances: overrides.max_instances || 1,
  ...overrides,
});

const catalog = {
  schema_version: 1,
  spec_schema_version: 2,
  graph_spec: {
    required_schema_version: 2,
    requires_explicit_route_fn: true,
    reserved_node_ids: ['START', 'END'],
    start_node: 'START',
    end_node: 'END',
  },
  node_catalog: {
    context_loader: node({
      type: 'context_loader',
      display_name: 'Context Loader',
      allowed_tool_contract_ids: ['thread_shape'],
      allowed_parent_types: ['START'],
      allowed_child_types: ['router', 'planner'],
    }),
    router: node({
      type: 'router',
      display_name: 'Router',
      allowed_route_functions: ['router_route'],
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['context_loader', 'hitl_gate'],
      allowed_child_types: ['retrieval_worker', 'direct_answer', 'finalizer', 'hitl_gate'],
    }),
    planner: node({
      type: 'planner',
      display_name: 'Planner',
      allowed_route_functions: ['planner_route'],
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['context_loader', 'hitl_gate'],
      allowed_child_types: ['retrieval_worker', 'finalizer', 'hitl_gate'],
    }),
    retrieval_worker: node({
      type: 'retrieval_worker',
      display_name: 'Document Retrieval',
      allowed_tool_contract_ids: ['document_evidence', 'focused_document_evidence'],
      allowed_parent_types: ['router', 'planner', 'replanner', 'hitl_gate'],
      allowed_child_types: ['evidence_evaluator', 'synthesizer', 'finalizer', 'hitl_gate'],
      max_instances: 4,
    }),
    evidence_evaluator: node({
      type: 'evidence_evaluator',
      display_name: 'Evidence Evaluator',
      allowed_route_functions: ['evaluator_route'],
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['retrieval_worker', 'hitl_gate'],
      allowed_child_types: ['synthesizer', 'replanner', 'hitl_gate'],
      max_instances: 2,
    }),
    replanner: node({
      type: 'replanner',
      display_name: 'Replanner',
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['evidence_evaluator', 'hitl_gate'],
      allowed_child_types: ['retrieval_worker', 'hitl_gate'],
    }),
    direct_answer: node({
      type: 'direct_answer',
      display_name: 'Direct Answer',
      allowed_parent_types: ['router', 'planner', 'hitl_gate'],
      allowed_child_types: ['finalizer', 'hitl_gate'],
    }),
    synthesizer: node({
      type: 'synthesizer',
      display_name: 'Synthesizer',
      allowed_parent_types: ['retrieval_worker', 'evidence_evaluator', 'hitl_gate'],
      allowed_child_types: ['finalizer', 'hitl_gate'],
    }),
    finalizer: node({
      type: 'finalizer',
      display_name: 'Finalizer',
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['router', 'planner', 'retrieval_worker', 'direct_answer', 'synthesizer', 'hitl_gate'],
      allowed_child_types: ['hitl_gate', 'END'],
    }),
    hitl_gate: node({
      type: 'hitl_gate',
      display_name: 'HITL Gate',
      allowed_route_functions: ['hitl_gate_route'],
      allowed_parent_types: ['router', 'planner', 'retrieval_worker', 'evidence_evaluator', 'synthesizer', 'finalizer'],
      allowed_child_types: ['retrieval_worker', 'synthesizer', 'finalizer', 'END'],
      max_instances: 8,
    }),
  },
  route_functions: {
    router_route: {
      allowed_source_types: ['router'],
      route_labels: ['document', 'memory', 'timeline', 'web', 'direct', 'clarify'],
    },
    planner_route: {
      allowed_source_types: ['planner'],
      route_labels: ['execute', 'direct', 'clarify'],
    },
    evaluator_route: {
      allowed_source_types: ['evidence_evaluator'],
      route_labels: ['answer', 'replan', 'answer_budget_exhausted'],
    },
    hitl_gate_route: {
      allowed_source_types: ['hitl_gate'],
      route_labels: null,
    },
  },
  tool_contracts: {
    thread_shape: {
      id: 'thread_shape',
      display_name: 'Thread Shape',
      canonical_tools: ['inspect_thread'],
      allowed_node_types: ['context_loader'],
      required_node_capabilities: [],
      artifact_keys: [],
      warning_codes: [],
    },
    clarify_intent: {
      id: 'clarify_intent',
      display_name: 'Clarify Intent',
      canonical_tools: ['clarify'],
      allowed_node_types: ['router', 'planner', 'finalizer', 'evidence_evaluator', 'replanner'],
      required_node_capabilities: [],
      artifact_keys: [],
      warning_codes: [],
    },
    document_evidence: {
      id: 'document_evidence',
      display_name: 'Document Evidence',
      canonical_tools: ['search_documents'],
      allowed_node_types: ['retrieval_worker'],
      required_node_capabilities: [],
      artifact_keys: ['document_sources'],
      warning_codes: [],
    },
    focused_document_evidence: {
      id: 'focused_document_evidence',
      display_name: 'Focused Document Evidence',
      canonical_tools: ['focused_search'],
      allowed_node_types: ['retrieval_worker'],
      required_node_capabilities: [],
      artifact_keys: ['document_sources'],
      warning_codes: [],
    },
  },
  defaults: {
    context_policy: {
      evidence_packet_limit: 12,
      evidence_packet_content_limit: 2000,
      final_prompt_assembly: 'legacy_evidence',
    },
    loop_policy: { default_max_node_visits: 1 },
  },
};

test('creates a router starter spec with canonical node ids and route function metadata', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const spec = assembleAgentPatternSpec(state);

  assert.deepEqual(state.nodes.map((item) => item.id), [
    'context_loader',
    'router',
    'retrieval_worker',
    'synthesizer',
    'finalizer',
  ]);
  assert.equal(spec.schema_version, 2);
  assert.equal(spec.pattern_type, 'custom_rag_agent');
  assert.deepEqual(spec.config.allowed_tool_ids, ['clarify_intent', 'document_evidence', 'thread_shape']);
  assert.equal(spec.config.graph.edges.find((edge) => edge.from === 'router')?.route_fn, 'router_route');
  assert.deepEqual(spec.config.graph.edges.find((edge) => edge.from === 'router')?.routes, {
    document: 'retrieval_worker',
    direct: 'finalizer',
    clarify: 'finalizer',
  });
});

test('catalog helpers filter routes, labels, tool contracts, and edge compatibility', () => {
  const state = createInitialBuilderState(catalog, 'router');

  assert.deepEqual(getAllowedRouteFunctionsForNode(catalog, 'router'), ['router_route']);
  assert.deepEqual(getRouteLabelsForFunction(catalog, 'planner_route'), ['execute', 'direct', 'clarify']);
  assert.deepEqual(
    getAllowedToolContractsForNode(catalog, 'retrieval_worker').map((contract) => contract.id),
    ['document_evidence', 'focused_document_evidence'],
  );
  assert.equal(canConnectNodes(catalog, state, 'router', 'retrieval_worker').ok, true);
  assert.equal(canConnectNodes(catalog, state, 'retrieval_worker', 'router').ok, false);
});

test('enforces catalog max instances and canonical fallback ids', () => {
  const state = createInitialBuilderState(catalog, 'router');

  assert.equal(canAddNodeType(catalog, state, 'router').ok, false);
  assert.equal(canAddNodeType(catalog, state, 'retrieval_worker').ok, true);
  assert.equal(getCanonicalNodeId('retrieval_worker', state.nodes.map((item) => item.id)), 'retrieval_worker_2');
});

test('loads a saved spec back into builder state and round-trips unchanged graph data', () => {
  const original = assembleAgentPatternSpec(createInitialBuilderState(catalog, 'plan_execute'));
  const loaded = loadBuilderStateFromSpec(original);
  const roundTrip = assembleAgentPatternSpec(loaded);

  assert.deepEqual(roundTrip.config.graph, original.config.graph);
  assert.deepEqual(roundTrip.config.allowed_tool_ids, original.config.allowed_tool_ids);
  assert.equal(roundTrip.config.graph.edges.find((edge) => edge.from === 'planner')?.route_fn, 'planner_route');
});

test('generates HITL gate nodes with matching conditional route and policy entries', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const gated = createHitlGateForTarget(catalog, state, 'retrieval_worker', {
    id: 'review_retrieval',
    sourceNodeId: 'router',
    title: 'Review retrieval',
    defaultAction: 'continue_without',
  });
  const spec = assembleAgentPatternSpec(gated);

  assert.equal(gated.nodes.find((item) => item.id === 'review_retrieval')?.type, 'hitl_gate');
  assert.equal(spec.config.hitl_policy.enabled, true);
  assert.deepEqual(spec.config.hitl_policy.gates.review_retrieval.target, { node_id: 'retrieval_worker' });
  assert.equal(
    spec.config.graph.edges.find((edge) => edge.from === 'review_retrieval')?.route_fn,
    'hitl_gate_route',
  );
  assert.deepEqual(spec.config.graph.edges.find((edge) => edge.from === 'review_retrieval')?.routes, {
    approve: 'retrieval_worker',
    continue_without: 'retrieval_worker',
    reject: 'END',
  });
});

test('normalizes unsupported node tools and over-limit node types from loaded state', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const normalized = normalizeBuilderState(catalog, {
    ...state,
    nodes: [
      ...state.nodes,
      { id: 'router_2', type: 'router', tool_contract_ids: ['document_evidence'] },
      { id: 'retrieval_worker_2', type: 'retrieval_worker', tool_contract_ids: ['missing_contract'] },
    ],
  });

  assert.equal(normalized.nodes.some((item) => item.id === 'router_2'), false);
  assert.equal(normalized.nodes.find((item) => item.id === 'retrieval_worker_2')?.tool_contract_ids, undefined);
  assert.deepEqual(normalized.allowed_tool_ids, ['clarify_intent', 'document_evidence', 'thread_shape']);
});

