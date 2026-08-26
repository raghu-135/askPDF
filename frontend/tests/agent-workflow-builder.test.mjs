import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  assembleAgentWorkflowSpec,
  canAddNodeType,
  canConnectNodes,
  canConnectNodeTypes,
  canConnectNodeTypeToTarget,
  canConnectSourceToType,
  canInsertExistingNodeBefore,
  canInsertNodeTypeBefore,
  createHitlGateForTarget,
  getIncomingPaths,
  getAgentWorkflowSourceKey,
  getImmediateSuccessorIds,
  getAllowedRouteFunctionsForNode,
  getAllowedToolContractsForNode,
  getCanonicalNodeId,
  getRouteLabelsForFunction,
  insertNodeBefore,
  isIsolatedBuilderNode,
  loadBuilderStateFromSpec,
  normalizeBuilderState,
  normalizeCorrectivePolicy,
  normalizeParallelPolicy,
  setHitlContinueWithoutTarget,
  wouldCreateBuilderCycle,
} from '../src/lib/agent-workflow-builder.ts';

test('uses the backend canonical key to load a canonical built-in workflow row', () => {
  assert.equal(getAgentWorkflowSourceKey({
    id: 'legacy-database-uuid',
    builtin_key: 'router_rag_agent',
    name: 'Router Agent',
    is_builtin: true,
  }), 'router_rag_agent');
});

const seededStarterSpecs = {
  router: JSON.parse(readFileSync(new URL('../../rag_service/app/agent_workflows/builtins/router_rag_agent.json', import.meta.url))).spec_json,
  plan_execute: JSON.parse(readFileSync(new URL('../../rag_service/app/agent_workflows/builtins/plan_execute_rag_agent.json', import.meta.url))).spec_json,
  evaluator_replanner: JSON.parse(readFileSync(new URL('../../rag_service/app/agent_workflows/builtins/evaluator_replanner_rag_agent.json', import.meta.url))).spec_json,
};

test('parallel workflow assembly enables bounded parallel runtime metadata', () => {
  const spec = assembleAgentWorkflowSpec({
    workflowId: 'parallel-custom',
    workflowType: 'custom_rag_agent',
    nodes: [
      { id: 'planner', type: 'planner' },
      { id: 'dispatch', type: 'parallel_dispatch' },
      { id: 'worker', type: 'retrieval_worker' },
      { id: 'aggregate', type: 'aggregator' },
    ],
    edges: [
      { from: 'planner', conditional: true, route_fn: 'planner_route', routes: { execute: 'dispatch' } },
      { from: 'dispatch', to: 'worker', dynamic: true },
      { from: 'dispatch', conditional: true, route_fn: 'parallel_dispatch_route', routes: { dispatch: 'aggregate' } },
      { from: 'worker', to: 'aggregate' },
    ],
    allowed_tool_ids: ['document_evidence'],
    parallel_policy: {
      enabled: true,
      max_concurrency: 4,
      max_work_items: 8,
      dispatch_timeout_ms: 60000,
      default_worker_timeout_ms: 30000,
      web_worker_timeout_ms: 45000,
      max_attempts: 2,
      minimum_successes: 1,
      continue_on_partial_failure: true,
    },
    runtime: { kind: 'compiled_rag', features: { supports_replans: false } },
  });

  assert.equal(spec.runtime.features.supports_parallel_dispatch, true);
  assert.equal(spec.config.parallel_policy.enabled, true);
  assert.equal(spec.config.parallel_policy.max_concurrency, 4);
  assert.equal(spec.config.graph.edges[1].dynamic, true);
});

const createInitialBuilderState = (currentCatalog, starter = 'router') => {
  return normalizeBuilderState(currentCatalog, loadBuilderStateFromSpec(seededStarterSpecs[starter]));
};

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
  spec_schema_version: 1,
  graph_spec: {
    required_schema_version: 1,
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
      allowed_child_types: ['router', 'planner', 'hitl_gate'],
    }),
    router: node({
      type: 'router',
      display_name: 'Router',
      allowed_route_functions: ['router_route'],
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['context_loader', 'hitl_gate'],
      allowed_child_types: ['planner', 'serial_dispatch', 'retrieval_worker', 'thread_conversation_history_worker', 'durable_memory_worker', 'thread_events_worker', 'direct_answer', 'finalizer', 'hitl_gate'],
    }),
    planner: node({
      type: 'planner',
      display_name: 'Planner',
      allowed_route_functions: ['planner_route'],
      allowed_tool_contract_ids: ['clarify_intent'],
      allowed_parent_types: ['context_loader', 'router', 'hitl_gate'],
      allowed_child_types: ['serial_dispatch', 'retrieval_worker', 'finalizer', 'hitl_gate'],
    }),
    retrieval_worker: node({
      type: 'retrieval_worker',
      display_name: 'Document Retrieval',
      allowed_tool_contract_ids: ['document_evidence', 'focused_document_evidence'],
      allowed_parent_types: ['router', 'planner', 'replanner', 'serial_dispatch', 'hitl_gate'],
      allowed_child_types: ['serial_dispatch', 'evidence_evaluator', 'synthesizer', 'finalizer', 'hitl_gate'],
      max_instances: 4,
    }),
    thread_conversation_history_worker: node({
      type: 'thread_conversation_history_worker',
      display_name: 'Thread Conversation History Retrieval',
      allowed_tool_contract_ids: ['thread_conversation_history'],
      allowed_parent_types: ['router'],
      allowed_child_types: ['synthesizer'],
    }),
    durable_memory_worker: node({
      type: 'durable_memory_worker',
      display_name: 'Durable Memory Retrieval',
      allowed_tool_contract_ids: ['durable_memory'],
      allowed_parent_types: ['router'],
      allowed_child_types: ['synthesizer'],
    }),
    thread_events_worker: node({
      type: 'thread_events_worker',
      display_name: 'Thread Events Retrieval',
      allowed_tool_contract_ids: ['thread_events'],
      allowed_parent_types: ['router'],
      allowed_child_types: ['synthesizer'],
    }),
    web_worker: node({
      type: 'web_worker',
      display_name: 'Web Retrieval',
      allowed_tool_contract_ids: ['live_web_recon'],
      allowed_parent_types: ['router'],
      allowed_child_types: ['synthesizer'],
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
      allowed_parent_types: ['retrieval_worker', 'thread_conversation_history_worker', 'durable_memory_worker', 'thread_events_worker', 'web_worker', 'evidence_evaluator', 'hitl_gate'],
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
      allowed_parent_types: ['START', 'context_loader', 'router', 'planner', 'retrieval_worker', 'evidence_evaluator', 'synthesizer', 'aggregator', 'answer_evaluator', 'finalizer'],
      allowed_child_types: ['router', 'planner', 'serial_dispatch', 'retrieval_worker', 'synthesizer', 'answer_evaluator', 'finalizer', 'END'],
      max_instances: 8,
    }),
    serial_dispatch: node({
      type: 'serial_dispatch',
      display_name: 'Serial Dispatch',
      allowed_route_functions: ['serial_dispatch_route'],
      allowed_parent_types: ['router', 'planner', 'replanner', 'retrieval_worker', 'thread_conversation_history_worker', 'durable_memory_worker', 'thread_events_worker', 'web_worker', 'hitl_gate'],
      allowed_child_types: ['retrieval_worker', 'thread_conversation_history_worker', 'durable_memory_worker', 'thread_events_worker', 'web_worker', 'aggregator'],
    }),
    aggregator: node({
      type: 'aggregator',
      display_name: 'Result Aggregator',
      allowed_parent_types: ['serial_dispatch', 'retrieval_worker', 'thread_conversation_history_worker', 'durable_memory_worker', 'thread_events_worker', 'web_worker'],
      allowed_child_types: ['evidence_evaluator', 'synthesizer', 'hitl_gate'],
      max_instances: 2,
    }),
    answer_evaluator: node({
      type: 'answer_evaluator',
      display_name: 'Answer Quality Review',
      allowed_route_functions: ['answer_quality_route'],
      allowed_parent_types: ['direct_answer', 'synthesizer', 'answer_reviser'],
      allowed_child_types: ['answer_reviser', 'finalizer'],
      max_instances: 2,
    }),
    answer_reviser: node({
      type: 'answer_reviser',
      display_name: 'Answer Reviser',
      allowed_parent_types: ['answer_evaluator'],
      allowed_child_types: ['answer_evaluator'],
    }),
  },
  route_functions: {
    router_route: {
      allowed_source_types: ['router'],
      route_labels: ['document', 'thread_conversation_history', 'durable_memory', 'thread_events', 'web', 'compound', 'direct', 'clarify'],
      target_types_by_label: {
        document: ['retrieval_worker', 'serial_dispatch'],
        thread_conversation_history: ['thread_conversation_history_worker', 'serial_dispatch'],
        durable_memory: ['durable_memory_worker', 'serial_dispatch'],
        thread_events: ['thread_events_worker', 'serial_dispatch'],
        web: ['web_worker', 'serial_dispatch'],
        compound: ['planner'],
        direct: ['direct_answer'],
        clarify: ['finalizer'],
      },
    },
    planner_route: {
      allowed_source_types: ['planner'],
      route_labels: ['execute', 'direct', 'clarify'],
      target_types_by_label: {
        execute: ['serial_dispatch', 'retrieval_worker', 'thread_conversation_history_worker', 'thread_events_worker', 'web_worker'],
        direct: ['direct_answer', 'finalizer'],
        clarify: ['finalizer'],
      },
    },
    evaluator_route: {
      allowed_source_types: ['evidence_evaluator'],
      route_labels: ['answer', 'replan', 'answer_budget_exhausted'],
      target_types_by_label: {
        answer: ['synthesizer'],
        replan: ['replanner'],
        answer_budget_exhausted: ['synthesizer'],
      },
    },
    hitl_gate_route: {
      allowed_source_types: ['hitl_gate'],
      route_labels: null,
      target_types_by_label: null,
    },
    serial_dispatch_route: { allowed_source_types: ['serial_dispatch'], route_labels: null, target_types_by_label: null },
    answer_quality_route: {
      allowed_source_types: ['answer_evaluator'],
      route_labels: ['pass', 'revise', 'finalize_cautious'],
      target_types_by_label: { pass: ['finalizer'], revise: ['answer_reviser'], finalize_cautious: ['finalizer'] },
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
    thread_conversation_history: {
      id: 'thread_conversation_history',
      display_name: 'Thread Conversation History',
      canonical_tools: ['search_thread_conversation_history'],
      allowed_node_types: ['thread_conversation_history_worker'],
      required_node_capabilities: [],
      artifact_keys: ['memory_sources'],
      warning_codes: [],
    },
    durable_memory: {
      id: 'durable_memory',
      display_name: 'Durable Memory',
      canonical_tools: ['search_durable_memory'],
      allowed_node_types: ['durable_memory_worker'],
      required_node_capabilities: [],
      artifact_keys: ['memory_refs'],
      warning_codes: [],
    },
    thread_events: {
      id: 'thread_events',
      display_name: 'Thread Events',
      canonical_tools: ['search_thread_events'],
      allowed_node_types: ['thread_events_worker'],
      required_node_capabilities: [],
      artifact_keys: ['timeline_sources'],
      warning_codes: [],
    },
    live_web_recon: {
      id: 'live_web_recon',
      display_name: 'Live Web Recon',
      canonical_tools: ['search_web'],
      allowed_node_types: ['web_worker'],
      required_node_capabilities: [],
      artifact_keys: ['web_sources'],
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
    parallel_policy: {
      defaults: {
        enabled: true, max_concurrency: 4, max_work_items: 8, dispatch_timeout_ms: 60000,
        default_worker_timeout_ms: 30000, web_worker_timeout_ms: 45000, max_attempts: 2,
        minimum_successes: 1, continue_on_partial_failure: true,
      },
      fields: {
        enabled: { type: 'boolean', default: true, label: 'Parallel execution' },
        max_concurrency: { type: 'integer', default: 4, minimum: 1, maximum: 16, step: 1, label: 'Maximum concurrency' },
        max_work_items: { type: 'integer', default: 8, minimum: 1, maximum: 32, step: 1, label: 'Maximum work items' },
        dispatch_timeout_ms: { type: 'integer', default: 60000, minimum: 1000, maximum: 300000, step: 1000, unit: 'ms', label: 'Dispatch timeout' },
        default_worker_timeout_ms: { type: 'integer', default: 30000, minimum: 1000, maximum: 120000, step: 1000, unit: 'ms', label: 'Worker timeout' },
        web_worker_timeout_ms: { type: 'integer', default: 45000, minimum: 1000, maximum: 180000, step: 1000, unit: 'ms', label: 'Web worker timeout' },
        max_attempts: { type: 'integer', default: 2, minimum: 1, maximum: 5, step: 1, label: 'Maximum attempts' },
        minimum_successes: { type: 'integer', default: 1, minimum: 1, maximum: 32, step: 1, label: 'Minimum successes' },
        continue_on_partial_failure: { type: 'boolean', default: true, label: 'Continue with partial evidence' },
      },
    },
    corrective_policy: {
      defaults: {
        minimum_relevance_confidence: 0.65, max_corrective_waves: 2,
        max_total_work_items: 8, max_total_tool_attempts: 12,
        allow_web_fallback: true, memory_evidence_mode: 'policy_scoped',
      },
      fields: {
        minimum_relevance_confidence: { type: 'number', default: 0.65, minimum: 0, maximum: 1, label: 'Relevance' },
        max_corrective_waves: { type: 'integer', default: 2, minimum: 1, maximum: 3, label: 'Waves' },
        max_total_work_items: { type: 'integer', default: 8, minimum: 2, maximum: 16, label: 'Work' },
        max_total_tool_attempts: { type: 'integer', default: 12, minimum: 2, maximum: 24, label: 'Attempts' },
        allow_web_fallback: { type: 'boolean', default: true, label: 'Web' },
        memory_evidence_mode: { type: 'enum', default: 'policy_scoped', values: ['disabled', 'policy_scoped'], label: 'Memory' },
      },
    },
  },
};

test('creates a router starter spec with canonical node ids and route function metadata', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const spec = assembleAgentWorkflowSpec(state);

  assert.deepEqual(state.nodes.map((item) => item.id), [
    'context_loader',
    'router',
    'planner',
    'serial_dispatch',
    'retrieval_worker',
    'thread_conversation_history_worker',
    'durable_memory_worker',
    'thread_events_worker',
    'web_worker',
    'direct_answer',
    'synthesizer',
    'aggregator',
    'answer_evaluator',
    'answer_reviser',
    'finalizer',
  ]);
  assert.equal(spec.schema_version, 1);
  assert.equal(spec.workflow_id, 'router_rag_agent');
  assert.equal(spec.workflow_type, 'custom_rag_agent');
  assert.deepEqual(spec.config.allowed_tool_ids, ['thread_shape', 'document_evidence', 'thread_conversation_history', 'durable_memory', 'thread_events', 'live_web_recon', 'clarify_intent']);
  assert.equal(spec.config.graph.edges.find((edge) => edge.from === 'router')?.route_fn, 'router_route');
  assert.deepEqual(spec.config.graph.edges.find((edge) => edge.from === 'router')?.routes, {
    document: 'serial_dispatch',
    thread_conversation_history: 'serial_dispatch',
    durable_memory: 'serial_dispatch',
    thread_events: 'serial_dispatch',
    web: 'serial_dispatch',
    compound: 'planner',
    direct: 'direct_answer',
    clarify: 'finalizer',
  });
});

test('catalog helpers filter routes, labels, tool contracts, and edge compatibility', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const parentIncompatibleCatalog = structuredClone(catalog);
  parentIncompatibleCatalog.node_catalog.retrieval_worker.allowed_parent_types = ['planner'];

  assert.deepEqual(getAllowedRouteFunctionsForNode(catalog, 'router'), ['router_route']);
  assert.deepEqual(getRouteLabelsForFunction(catalog, 'planner_route'), ['execute', 'direct', 'clarify']);
  assert.deepEqual(
    getAllowedToolContractsForNode(catalog, 'retrieval_worker').map((contract) => contract.id),
    ['document_evidence', 'focused_document_evidence'],
  );
  assert.equal(canConnectNodes(catalog, state, 'router', 'retrieval_worker').ok, true);
  assert.equal(canConnectNodes(catalog, state, 'retrieval_worker', 'router').ok, false);
  assert.equal(canConnectNodes(catalog, state, 'START', 'retrieval_worker').ok, false);
  assert.equal(canConnectNodes(catalog, state, 'planner', 'END').ok, false);
  assert.equal(canConnectNodeTypes(parentIncompatibleCatalog, 'router', 'retrieval_worker').ok, false);
  assert.equal(canConnectSourceToType(parentIncompatibleCatalog, state, 'router', 'retrieval_worker').ok, false);
  const evaluatorTargets = catalog.route_functions.evaluator_route.target_types_by_label;
  assert.deepEqual(evaluatorTargets.answer, ['synthesizer']);
  assert.deepEqual(evaluatorTargets.replan, ['replanner']);
});

test('enforces catalog max instances and canonical fallback ids', () => {
  const state = createInitialBuilderState(catalog, 'router');

  assert.equal(canAddNodeType(catalog, state, 'router').ok, false);
  assert.equal(canAddNodeType(catalog, state, 'retrieval_worker').ok, true);
  assert.equal(getCanonicalNodeId('retrieval_worker', state.nodes.map((item) => item.id)), 'retrieval_worker_2');
});

test('loads a saved spec back into builder state and round-trips unchanged graph data', () => {
  const original = assembleAgentWorkflowSpec(createInitialBuilderState(catalog, 'plan_execute'));
  const loaded = loadBuilderStateFromSpec(original);
  const roundTrip = assembleAgentWorkflowSpec(loaded);

  assert.deepEqual(roundTrip.config.graph, original.config.graph);
  assert.deepEqual(roundTrip.config.allowed_tool_ids, original.config.allowed_tool_ids);
  assert.equal(roundTrip.workflow_id, original.workflow_id);
  assert.equal(roundTrip.config.graph.edges.find((edge) => edge.from === 'planner')?.route_fn, 'planner_route');
});

test('hydrates built-in node tool assignments from the global allowed tool list', () => {
  for (const starter of ['router', 'plan_execute', 'evaluator_replanner']) {
    const expectedToolIds = seededStarterSpecs[starter].config.allowed_tool_ids.filter((id) => catalog.tool_contracts[id]);
    const state = createInitialBuilderState(catalog, starter);

    assert.deepEqual([...state.allowed_tool_ids].sort(), [...expectedToolIds].sort());
    assert.deepEqual(
      state.nodes.find((item) => item.type === 'durable_memory_worker')?.tool_contract_ids,
      ['durable_memory'],
    );
  }
});

test('keeps every default starter loop budget within its per-node visit limits', () => {
  for (const starter of ['router', 'plan_execute', 'evaluator_replanner']) {
    const spec = assembleAgentWorkflowSpec(createInitialBuilderState(catalog, starter));
    const nodes = spec.config.graph.nodes;
    const policy = spec.config.loop_policy;
    const effectiveTotal = nodes.reduce(
      (total, item) => total + (policy.node_visit_limits[item.id] || policy.default_max_node_visits),
      0,
    );

    assert.ok(spec.workflow_id);
    assert.ok(policy.max_total_visits >= nodes.length);
    assert.ok(policy.max_total_visits <= effectiveTotal);
  }
});

test('stores canvas positions and notes as builder-only metadata', () => {
  const state = createInitialBuilderState(catalog, 'router');
  state.nodes[0].position = { x: 120, y: 80 };
  state.builder_ui = {
    notes: [{ id: 'note-1', text: 'Review this branch', position: { x: 20, y: 30 } }],
  };
  const spec = assembleAgentWorkflowSpec(state);

  assert.equal('position' in spec.config.graph.nodes[0], false);
  assert.deepEqual(spec.config.builder_ui.positions[state.nodes[0].id], { x: 120, y: 80 });
  assert.equal(spec.config.graph.nodes.some((item) => item.id === 'note-1'), false);

  const loaded = loadBuilderStateFromSpec(spec);
  assert.deepEqual(loaded.nodes[0].position, { x: 120, y: 80 });
  assert.equal(loaded.builder_ui.notes[0].text, 'Review this branch');
});

test('generates HITL gate nodes with matching conditional route and policy entries', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const incomingPath = getIncomingPaths(state, 'serial_dispatch')
    .find((path) => path.source === 'router' && path.route === 'document');
  const gated = createHitlGateForTarget(catalog, state, 'serial_dispatch', {
    id: 'review_retrieval',
    incomingPath,
    title: 'Review retrieval',
    defaultAction: 'approve',
  });
  const spec = assembleAgentWorkflowSpec(gated);

  assert.equal(gated.nodes.find((item) => item.id === 'review_retrieval')?.type, 'hitl_gate');
  assert.equal(spec.config.hitl_policy.enabled, true);
  assert.deepEqual(spec.config.hitl_policy.gates.review_retrieval.target, { node_id: 'serial_dispatch' });
  assert.equal(
    gated.edges.find((edge) => edge.from === 'router')?.routes?.document,
    'review_retrieval',
  );
  assert.equal(
    spec.config.graph.edges.find((edge) => edge.from === 'review_retrieval')?.route_fn,
    'hitl_gate_route',
  );
  assert.deepEqual(spec.config.graph.edges.find((edge) => edge.from === 'review_retrieval')?.routes, {
    approve: 'serial_dispatch',
    reject: 'END',
  });
});

test('inserts a HITL gate between context loading and planning as a compatible path step', () => {
  const state = createInitialBuilderState(catalog, 'plan_execute');
  const incomingPath = getIncomingPaths(state, 'planner')[0];
  const gated = createHitlGateForTarget(catalog, state, 'planner', {
    id: 'review_planner',
    incomingPath,
  });

  assert.equal(canConnectNodes(catalog, gated, 'context_loader', 'review_planner').ok, true);
  assert.equal(canConnectNodes(catalog, gated, 'review_planner', 'planner').ok, true);
  assert.equal(
    gated.edges.some((edge) => edge.from === 'context_loader' && edge.to === 'review_planner'),
    true,
  );
});

test('resolves HITL bypasses only for a unique immediate successor', () => {
  const sequential = createInitialBuilderState(catalog, 'router');
  assert.deepEqual(getImmediateSuccessorIds(sequential, 'retrieval_worker'), ['serial_dispatch']);
  const retrievalGate = createHitlGateForTarget(catalog, sequential, 'retrieval_worker', { id: 'review_retrieval' });
  const retrievalRoutes = retrievalGate.edges.find((edge) => edge.from === 'review_retrieval')?.routes;
  assert.equal(retrievalRoutes.approve, 'retrieval_worker');
  assert.equal(retrievalRoutes.continue_without, 'serial_dispatch');

  const routerGate = createHitlGateForTarget(catalog, sequential, 'router', { id: 'review_router' });
  const routerNode = routerGate.nodes.find((item) => item.id === 'review_router');
  const routerRoutes = routerGate.edges.find((edge) => edge.from === 'review_router')?.routes;
  assert.equal('continue_without' in routerRoutes, false);
  assert.deepEqual(routerNode.hitl.allowed_actions, ['approve', 'reject']);
  assert.equal(routerNode.hitl.default_action, 'approve');
});

test('updates and clears a HITL continue-without target atomically', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const gated = createHitlGateForTarget(catalog, state, 'router', { id: 'review_router' });
  const enabled = setHitlContinueWithoutTarget(gated, 'review_router', 'serial_dispatch');
  const enabledNode = enabled.nodes.find((item) => item.id === 'review_router');
  const enabledEdge = enabled.edges.find((edge) => edge.from === 'review_router');
  assert.equal(enabledNode.hitl.routes.continue_without, 'serial_dispatch');
  assert.equal(enabledEdge.routes.continue_without, 'serial_dispatch');
  assert.equal(enabledNode.hitl.allowed_actions.includes('continue_without'), true);

  enabledNode.hitl.default_action = 'continue_without';
  const cleared = setHitlContinueWithoutTarget(enabled, 'review_router');
  const clearedNode = cleared.nodes.find((item) => item.id === 'review_router');
  const clearedEdge = cleared.edges.find((edge) => edge.from === 'review_router');
  assert.equal('continue_without' in clearedNode.hitl.routes, false);
  assert.equal('continue_without' in clearedEdge.routes, false);
  assert.equal(clearedNode.hitl.allowed_actions.includes('continue_without'), false);
  assert.equal(clearedNode.hitl.default_action, 'approve');
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
  assert.deepEqual(
    [...normalized.allowed_tool_ids].sort(),
    [...seededStarterSpecs.router.config.allowed_tool_ids].filter((id) => catalog.tool_contracts[id]).sort(),
  );
});

test('expands sequential and conditional incoming paths separately', () => {
  const state = {
    ...createInitialBuilderState(catalog, 'router'),
    edges: [
      { from: 'retrieval_worker', to: 'finalizer' },
      {
        from: 'router',
        conditional: true,
        route_fn: 'router_route',
        routes: { document: 'finalizer', direct: 'finalizer', clarify: 'synthesizer' },
      },
    ],
  };
  const paths = getIncomingPaths(state, 'finalizer');

  assert.deepEqual(paths.map((path) => [path.source, path.route]), [
    ['retrieval_worker', undefined],
    ['router', 'document'],
    ['router', 'direct'],
  ]);
  assert.equal(new Set(paths.map((path) => path.id)).size, 3);
});

test('filters previous-step types using both sides and excludes route-producing nodes', () => {
  const state = {
    ...createInitialBuilderState(catalog, 'router'),
    nodes: createInitialBuilderState(catalog, 'router').nodes.filter((item) => item.type !== 'synthesizer'),
    edges: [{ from: 'retrieval_worker', to: 'finalizer' }],
  };
  const path = getIncomingPaths(state, 'finalizer')[0];

  assert.equal(canInsertNodeTypeBefore(catalog, state, 'finalizer', 'synthesizer', path).ok, true);
  assert.equal(canInsertNodeTypeBefore(catalog, state, 'finalizer', 'direct_answer', path).ok, false);
  assert.equal(canConnectNodeTypeToTarget(catalog, state, 'direct_answer', 'finalizer').ok, true);
  assert.match(canInsertNodeTypeBefore(catalog, state, 'finalizer', 'evidence_evaluator', path).reason, /named outgoing routes/);
});

test('inserts a node into one sequential path atomically', () => {
  const state = {
    ...createInitialBuilderState(catalog, 'router'),
    edges: [{ from: 'retrieval_worker', to: 'finalizer' }],
  };
  const path = getIncomingPaths(state, 'finalizer')[0];
  const inserted = insertNodeBefore(
    state,
    'finalizer',
    { id: 'synthesizer_2', type: 'synthesizer', position: { x: 10, y: 20 } },
    path,
  );

  assert.deepEqual(inserted.edges, [
    { from: 'retrieval_worker', to: 'synthesizer_2' },
    { from: 'synthesizer_2', to: 'finalizer' },
  ]);
  assert.deepEqual(inserted.nodes.at(-1)?.position, { x: 10, y: 20 });
  assert.deepEqual(
    loadBuilderStateFromSpec(assembleAgentWorkflowSpec(inserted)).edges,
    inserted.edges,
  );
});

test('rewires only the selected conditional route during insertion', () => {
  const state = {
    ...createInitialBuilderState(catalog, 'router'),
    edges: [{
      from: 'router',
      conditional: true,
      route_fn: 'router_route',
      routes: { document: 'finalizer', direct: 'finalizer' },
    }],
  };
  const path = getIncomingPaths(state, 'finalizer').find((candidate) => candidate.route === 'direct');
  const inserted = insertNodeBefore(state, 'finalizer', { id: 'direct_answer_2', type: 'direct_answer' }, path);

  assert.deepEqual(inserted.edges[0].routes, {
    document: 'finalizer',
    direct: 'direct_answer_2',
  });
  assert.deepEqual(inserted.edges[1], { from: 'direct_answer_2', to: 'finalizer' });
});

test('supports previous insertion with no incoming path and at START or END boundaries', () => {
  const noIncoming = {
    ...createInitialBuilderState(catalog, 'router'),
    edges: [],
  };
  const detached = insertNodeBefore(noIncoming, 'router', { id: 'context_loader_2', type: 'context_loader' });
  assert.deepEqual(detached.edges, [{ from: 'context_loader_2', to: 'router' }]);

  const startState = {
    ...noIncoming,
    nodes: noIncoming.nodes.filter((item) => item.type !== 'context_loader'),
    edges: [{ from: 'START', to: 'router' }],
  };
  const startPath = getIncomingPaths(startState, 'router')[0];
  assert.equal(canInsertNodeTypeBefore(catalog, startState, 'router', 'context_loader', startPath).ok, true);

  const endState = {
    ...noIncoming,
    nodes: noIncoming.nodes.filter((item) => item.type !== 'finalizer'),
    edges: [{ from: 'retrieval_worker', to: 'END' }],
  };
  const endPath = getIncomingPaths(endState, 'END')[0];
  assert.equal(canInsertNodeTypeBefore(catalog, endState, 'END', 'finalizer', endPath).ok, true);
});

test('offers only isolated existing nodes and detects paths that would form cycles', () => {
  const base = createInitialBuilderState(catalog, 'router');
  const state = {
    ...base,
    nodes: [...base.nodes, { id: 'synthesizer_2', type: 'synthesizer' }],
    edges: [{ from: 'retrieval_worker', to: 'finalizer' }],
  };
  const path = getIncomingPaths(state, 'finalizer')[0];

  assert.equal(isIsolatedBuilderNode(state, 'synthesizer_2'), true);
  assert.equal(canInsertExistingNodeBefore(catalog, state, 'finalizer', 'synthesizer_2', path).ok, true);
  assert.equal(canInsertExistingNodeBefore(catalog, state, 'finalizer', 'retrieval_worker', path).ok, false);

  const cyclic = {
    ...state,
    edges: [{ from: 'retrieval_worker', to: 'synthesizer_2' }, { from: 'synthesizer_2', to: 'finalizer' }],
  };
  assert.equal(wouldCreateBuilderCycle(cyclic, 'finalizer', 'retrieval_worker'), true);
});

test('blocks manual connections that bypass a parallel region', () => {
  const parallelCatalog = structuredClone(catalog);
  parallelCatalog.node_catalog.parallel_dispatch = node({
    type: 'parallel_dispatch',
    display_name: 'Parallel Dispatch',
    allowed_parent_types: ['planner'],
    allowed_child_types: ['retrieval_worker', 'aggregator'],
  });
  parallelCatalog.node_catalog.aggregator = node({
    type: 'aggregator',
    display_name: 'Aggregator',
    allowed_parent_types: ['parallel_dispatch', 'retrieval_worker'],
    allowed_child_types: ['synthesizer'],
  });
  parallelCatalog.node_catalog.planner.allowed_child_types.push('parallel_dispatch');
  parallelCatalog.node_catalog.retrieval_worker.allowed_parent_types.push('parallel_dispatch');
  parallelCatalog.node_catalog.retrieval_worker.allowed_child_types.push('aggregator');
  parallelCatalog.node_catalog.retrieval_worker.parallel_state_writes = ['worker_result_packets'];
  parallelCatalog.node_catalog.synthesizer.allowed_parent_types.push('aggregator');

  const state = {
    ...createInitialBuilderState(parallelCatalog, 'router'),
    nodes: [
      { id: 'planner', type: 'planner' },
      { id: 'dispatch', type: 'parallel_dispatch' },
      { id: 'worker', type: 'retrieval_worker' },
      { id: 'aggregate', type: 'aggregator' },
      { id: 'synthesizer', type: 'synthesizer' },
    ],
    edges: [],
  };

  assert.equal(canConnectNodes(parallelCatalog, state, 'dispatch', 'worker').ok, true);
  assert.equal(canConnectNodes(parallelCatalog, state, 'worker', 'aggregate').ok, true);
  assert.equal(canConnectNodes(parallelCatalog, state, 'planner', 'worker').ok, false);
  assert.equal(canConnectNodes(parallelCatalog, state, 'worker', 'synthesizer').ok, false);
});

test('normalizes catalog-driven parallel policy and removes it with the region', () => {
  const policy = normalizeParallelPolicy(catalog, {
    max_concurrency: 999,
    dispatch_timeout_ms: 1,
    max_work_items: 2,
    minimum_successes: 9,
    continue_on_partial_failure: false,
  });
  assert.equal(policy.max_concurrency, 16);
  assert.equal(policy.dispatch_timeout_ms, 1000);
  assert.equal(policy.minimum_successes, 2);
  assert.equal(policy.continue_on_partial_failure, false);

  const withoutRegion = normalizeBuilderState(catalog, {
    ...createInitialBuilderState(catalog, 'router'),
    parallel_policy: policy,
    runtime: { features: { supports_parallel_dispatch: true, supports_replans: false } },
  });
  assert.equal(withoutRegion.parallel_policy, undefined);
  assert.equal(withoutRegion.runtime.features.supports_parallel_dispatch, undefined);
});

test('round-trips only validated corrective policy controls', () => {
  const policy = normalizeCorrectivePolicy(catalog, {
    minimum_relevance_confidence: 9,
    max_corrective_waves: 99,
    max_total_work_items: 6,
    max_total_tool_attempts: 10,
    allow_web_fallback: false,
    memory_evidence_mode: 'invalid',
  });
  assert.equal(policy.minimum_relevance_confidence, 1);
  assert.equal(policy.max_corrective_waves, 3);
  assert.equal(policy.allow_web_fallback, false);
  assert.equal(policy.memory_evidence_mode, 'policy_scoped');

  const spec = assembleAgentWorkflowSpec({
    workflowId: 'corrective_self_rag_agent', workflowType: 'custom_rag_agent',
    nodes: [{ id: 'grader', type: 'retrieval_quality_grader' }, { id: 'verifier', type: 'grounded_answer_verifier' }],
    edges: [], allowed_tool_ids: [], corrective_policy: policy,
    runtime: { kind: 'compiled_rag', features: {} },
  });
  assert.deepEqual(spec.config.corrective_policy, policy);
});
