import assert from 'node:assert/strict';
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
  createInitialBuilderState,
  getIncomingPaths,
  getImmediateSuccessorIds,
  getAllowedRouteFunctionsForNode,
  getAllowedToolContractsForNode,
  getCanonicalNodeId,
  getRouteLabelsForFunction,
  insertNodeBefore,
  isIsolatedBuilderNode,
  loadBuilderStateFromSpec,
  normalizeBuilderState,
  setHitlContinueWithoutTarget,
  wouldCreateBuilderCycle,
} from '../src/lib/agent-workflow-builder.ts';

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
      allowed_child_types: ['router', 'planner', 'hitl_gate'],
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
    memory_worker: node({
      type: 'memory_worker',
      display_name: 'Memory Retrieval',
      allowed_tool_contract_ids: ['deep_memory'],
      allowed_parent_types: ['router'],
      allowed_child_types: ['synthesizer'],
    }),
    timeline_worker: node({
      type: 'timeline_worker',
      display_name: 'Timeline Retrieval',
      allowed_tool_contract_ids: ['thread_timeline'],
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
      allowed_parent_types: ['retrieval_worker', 'memory_worker', 'timeline_worker', 'web_worker', 'evidence_evaluator', 'hitl_gate'],
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
      allowed_parent_types: ['START', 'context_loader', 'router', 'planner', 'retrieval_worker', 'evidence_evaluator', 'synthesizer', 'finalizer'],
      allowed_child_types: ['router', 'planner', 'retrieval_worker', 'synthesizer', 'finalizer', 'END'],
      max_instances: 8,
    }),
  },
  route_functions: {
    router_route: {
      allowed_source_types: ['router'],
      route_labels: ['document', 'memory', 'timeline', 'web', 'direct', 'clarify'],
      target_types_by_label: {
        document: ['retrieval_worker'],
        memory: ['memory_worker'],
        timeline: ['timeline_worker'],
        web: ['web_worker'],
        direct: ['direct_answer'],
        clarify: ['finalizer'],
      },
    },
    planner_route: {
      allowed_source_types: ['planner'],
      route_labels: ['execute', 'direct', 'clarify'],
      target_types_by_label: {
        execute: ['retrieval_worker', 'memory_worker', 'timeline_worker', 'web_worker'],
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
    deep_memory: {
      id: 'deep_memory',
      display_name: 'Deep Memory',
      canonical_tools: ['search_memory'],
      allowed_node_types: ['memory_worker'],
      required_node_capabilities: [],
      artifact_keys: ['memory_sources'],
      warning_codes: [],
    },
    thread_timeline: {
      id: 'thread_timeline',
      display_name: 'Thread Timeline',
      canonical_tools: ['search_timeline'],
      allowed_node_types: ['timeline_worker'],
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
  },
};

test('creates a router starter spec with canonical node ids and route function metadata', () => {
  const state = createInitialBuilderState(catalog, 'router');
  const spec = assembleAgentWorkflowSpec(state);

  assert.deepEqual(state.nodes.map((item) => item.id), [
    'context_loader',
    'router',
    'retrieval_worker',
    'memory_worker',
    'long_term_memory_worker',
    'timeline_worker',
    'web_worker',
    'direct_answer',
    'synthesizer',
    'finalizer',
  ]);
  assert.equal(spec.schema_version, 2);
  assert.equal(spec.workflow_id, 'custom_rag_agent');
  assert.equal(spec.workflow_type, 'custom_rag_agent');
  assert.deepEqual(spec.config.allowed_tool_ids, ['clarify_intent', 'deep_memory', 'document_evidence', 'live_web_recon', 'thread_shape', 'thread_timeline']);
  assert.equal(spec.config.graph.edges.find((edge) => edge.from === 'router')?.route_fn, 'router_route');
  assert.deepEqual(spec.config.graph.edges.find((edge) => edge.from === 'router')?.routes, {
    document: 'retrieval_worker',
    memory: 'memory_worker',
    long_term_memory: 'long_term_memory_worker',
    timeline: 'timeline_worker',
    web: 'web_worker',
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
  const incomingPath = getIncomingPaths(state, 'retrieval_worker')
    .find((path) => path.source === 'router' && path.route === 'document');
  const gated = createHitlGateForTarget(catalog, state, 'retrieval_worker', {
    id: 'review_retrieval',
    incomingPath,
    title: 'Review retrieval',
    defaultAction: 'continue_without',
  });
  const spec = assembleAgentWorkflowSpec(gated);

  assert.equal(gated.nodes.find((item) => item.id === 'review_retrieval')?.type, 'hitl_gate');
  assert.equal(spec.config.hitl_policy.enabled, true);
  assert.deepEqual(spec.config.hitl_policy.gates.review_retrieval.target, { node_id: 'retrieval_worker' });
  assert.equal(
    gated.edges.find((edge) => edge.from === 'router')?.routes?.document,
    'review_retrieval',
  );
  assert.equal(
    spec.config.graph.edges.find((edge) => edge.from === 'review_retrieval')?.route_fn,
    'hitl_gate_route',
  );
  assert.deepEqual(spec.config.graph.edges.find((edge) => edge.from === 'review_retrieval')?.routes, {
    approve: 'retrieval_worker',
    reject: 'END',
    continue_without: 'synthesizer',
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
  assert.deepEqual(getImmediateSuccessorIds(sequential, 'retrieval_worker'), ['synthesizer']);
  const retrievalGate = createHitlGateForTarget(catalog, sequential, 'retrieval_worker', { id: 'review_retrieval' });
  const retrievalRoutes = retrievalGate.edges.find((edge) => edge.from === 'review_retrieval')?.routes;
  assert.equal(retrievalRoutes.approve, 'retrieval_worker');
  assert.equal(retrievalRoutes.continue_without, 'synthesizer');

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
  const enabled = setHitlContinueWithoutTarget(gated, 'review_router', 'retrieval_worker');
  const enabledNode = enabled.nodes.find((item) => item.id === 'review_router');
  const enabledEdge = enabled.edges.find((edge) => edge.from === 'review_router');
  assert.equal(enabledNode.hitl.routes.continue_without, 'retrieval_worker');
  assert.equal(enabledEdge.routes.continue_without, 'retrieval_worker');
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
  assert.deepEqual(normalized.allowed_tool_ids, [
    'clarify_intent',
    'deep_memory',
    'document_evidence',
    'live_web_recon',
    'thread_shape',
    'thread_timeline',
  ]);
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
