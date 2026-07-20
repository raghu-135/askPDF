import type {
  AgentWorkflowBuilderSpec,
  AgentWorkflowCatalogResponse,
  AgentWorkflowGraphSpec,
  AgentWorkflowNodeCatalogEntry,
  AgentWorkflowRouteFunctionMetadata,
  AgentWorkflowToolContract,
} from './api';
import { AgentRunResumeAction, BuiltinAgentNodeType, HitlMode, HitlPhase, RouteFunctionId } from './enums.ts';

export type AgentWorkflowStarter = 'router' | 'plan_execute' | 'evaluator_replanner';

export interface BuilderNodeState {
  id: string;
  type: string;
  label?: string;
  description?: string;
  tool_contract_ids?: string[];
  hitl?: {
    title?: string;
    body?: string;
    prompt?: string;
    phase?: string;
    mode?: string;
    allowed_actions?: string[];
    default_action?: string;
    routes?: Record<string, string>;
  };
  [key: string]: any;
}

export interface BuilderEdgeState {
  from: string;
  to?: string;
  conditional?: boolean;
  route_fn?: string;
  routes?: Record<string, string>;
  [key: string]: any;
}

export interface AgentWorkflowBuilderState {
  name?: string;
  description?: string;
  workflowId?: string;
  workflowType: string;
  nodes: BuilderNodeState[];
  edges: BuilderEdgeState[];
  allowed_tool_ids: string[];
  context_policy?: Record<string, any>;
  loop_policy?: Record<string, any>;
  hitl_policy?: Record<string, any>;
  runtime?: Record<string, any>;
  extraConfig?: Record<string, any>;
  builder_ui?: {
    notes?: { id: string; text: string; position: { x: number; y: number } }[];
    groups?: { id: string; label: string; node_ids: string[]; position?: { x: number; y: number } }[];
  };
}

export interface CompatibilityResult {
  ok: boolean;
  reason?: string;
}

export interface BuilderIncomingPath {
  id: string;
  edgeIndex: number;
  source: string;
  target: string;
  route?: string;
  conditional: boolean;
}

const ROUTE_FUNCTION_BY_NODE_TYPE: Record<string, string> = {
  [BuiltinAgentNodeType.Router]: RouteFunctionId.Router,
  [BuiltinAgentNodeType.Planner]: RouteFunctionId.Planner,
  [BuiltinAgentNodeType.EvidenceEvaluator]: RouteFunctionId.Evaluator,
  [BuiltinAgentNodeType.HitlGate]: RouteFunctionId.HitlGate,
};

const REPEATABLE_NODE_TYPES = new Set([
  'retrieval_worker',
  'memory_worker',
  'timeline_worker',
  'web_worker',
  'evidence_evaluator',
]);

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));

const defaultRuntime = (supportsReplans = false, promptPreview = 'router') => ({
  kind: 'compiled_rag',
  label: 'Compiled RAG',
  failure_code: 'compiled_rag_execution_failed',
  failure_reason_prefix: 'Exception during compiled RAG execution',
  success_context: 'Context retrieved by compiled RAG workflow.',
  failure_context: 'Compiled RAG workflow execution failed gracefully.',
  features: { supports_replans: supportsReplans },
  prompt_preview: promptPreview,
});

const nodeTypeById = (nodes: BuilderNodeState[]) => (
  new Map(nodes.map((node) => [node.id, node.type]))
);

const getNode = (state: AgentWorkflowBuilderState, nodeId: string) => (
  state.nodes.find((node) => node.id === nodeId)
);

const catalogEntry = (
  catalog: AgentWorkflowCatalogResponse,
  nodeType?: string,
): AgentWorkflowNodeCatalogEntry | undefined => (
  nodeType ? catalog.node_catalog[nodeType] : undefined
);

const routeMetadata = (
  catalog: AgentWorkflowCatalogResponse,
  routeFn?: string,
): AgentWorkflowRouteFunctionMetadata | undefined => (
  routeFn ? catalog.route_functions[routeFn] : undefined
);

export function getCanonicalNodeId(nodeType: string, existingIds: Iterable<string> = []): string {
  const existing = new Set(existingIds);
  if (!existing.has(nodeType)) return nodeType;
  for (let index = 2; index < 100; index += 1) {
    const candidate = `${nodeType}_${index}`;
    if (!existing.has(candidate)) return candidate;
  }
  return `${nodeType}_${Date.now()}`;
}

export function getAllowedToolContractsForNode(
  catalog: AgentWorkflowCatalogResponse,
  nodeType: string,
): AgentWorkflowToolContract[] {
  const allowedIds = new Set(catalogEntry(catalog, nodeType)?.allowed_tool_contract_ids || []);
  return Object.values(catalog.tool_contracts || {})
    .filter((contract) => allowedIds.has(contract.id))
    .sort((a, b) => (a.display_name || a.id).localeCompare(b.display_name || b.id));
}

export function getAllowedRouteFunctionsForNode(
  catalog: AgentWorkflowCatalogResponse,
  nodeType: string,
): string[] {
  const entry = catalogEntry(catalog, nodeType);
  const allowed = entry?.allowed_route_functions || [];
  return allowed.filter((routeFn) => {
    const metadata = routeMetadata(catalog, routeFn);
    const sourceTypes = metadata?.allowed_source_types || metadata?.allowed_source_node_types || [];
    return sourceTypes.length === 0 || sourceTypes.includes(nodeType);
  });
}

export function getRouteLabelsForFunction(
  catalog: AgentWorkflowCatalogResponse,
  routeFn: string,
): string[] | null {
  const metadata = routeMetadata(catalog, routeFn);
  const labels = metadata?.route_labels ?? metadata?.routes;
  return Array.isArray(labels) ? labels.filter((label): label is string => typeof label === 'string' && label.length > 0) : null;
}

export function getDefaultRouteFunctionForNode(
  catalog: AgentWorkflowCatalogResponse,
  nodeType: string,
): string | undefined {
  const preferred = ROUTE_FUNCTION_BY_NODE_TYPE[nodeType];
  const allowed = getAllowedRouteFunctionsForNode(catalog, nodeType);
  return preferred && allowed.includes(preferred) ? preferred : allowed[0];
}

export function canConnectNodeTypes(
  catalog: AgentWorkflowCatalogResponse,
  sourceType: string | undefined,
  targetType: string | undefined,
): CompatibilityResult {
  if (!sourceType || !targetType) return { ok: false, reason: 'Unknown source or target node type.' };
  const source = catalogEntry(catalog, sourceType);
  if (!source) return { ok: false, reason: `Unknown source node type: ${sourceType}` };
  if (!catalogEntry(catalog, targetType)) return { ok: false, reason: `Unknown target node type: ${targetType}` };
  if (!(source.allowed_child_types || []).includes(targetType)) {
    return { ok: false, reason: `${sourceType} cannot connect to ${targetType}.` };
  }
  return { ok: true };
}

export function canConnectNodes(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  sourceId: string,
  targetId: string,
): CompatibilityResult {
  if (sourceId === 'START') {
    const target = getNode(state, targetId);
    const allowedParents = catalogEntry(catalog, target?.type)?.allowed_parent_types || [];
    return allowedParents.includes('START')
      ? { ok: true }
      : { ok: false, reason: `${target?.type || targetId} cannot start the graph.` };
  }
  if (targetId === 'END') {
    const source = getNode(state, sourceId);
    const allowedChildren = catalogEntry(catalog, source?.type)?.allowed_child_types || [];
    return allowedChildren.includes('END')
      ? { ok: true }
      : { ok: false, reason: `${source?.type || sourceId} cannot end the graph.` };
  }
  const types = nodeTypeById(state.nodes);
  return canConnectNodeTypes(catalog, types.get(sourceId), types.get(targetId));
}

export function canAddNodeType(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  nodeType: string,
): CompatibilityResult {
  const entry = catalogEntry(catalog, nodeType);
  if (!entry) return { ok: false, reason: `Unknown node type: ${nodeType}` };
  const count = state.nodes.filter((node) => node.type === nodeType).length;
  if (count >= entry.max_instances) {
    return { ok: false, reason: `${nodeType} allows at most ${entry.max_instances} instance(s).` };
  }
  return { ok: true };
}

export function getCompatibleTargetNodes(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  sourceId: string,
): BuilderNodeState[] {
  return state.nodes.filter((node) => canConnectNodes(catalog, state, sourceId, node.id).ok);
}

export function getCompatibleSourceNodes(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  targetId: string,
): BuilderNodeState[] {
  return state.nodes.filter((node) => canConnectNodes(catalog, state, node.id, targetId).ok);
}

export function getIncomingPaths(
  state: AgentWorkflowBuilderState,
  targetId: string,
): BuilderIncomingPath[] {
  return state.edges.flatMap((edge, edgeIndex): BuilderIncomingPath[] => {
    if (edge.conditional) {
      return Object.entries(edge.routes || {})
        .filter(([, target]) => target === targetId)
        .map(([route]) => ({
          id: `${edgeIndex}:${edge.from}:${route}:${targetId}`,
          edgeIndex,
          source: edge.from,
          target: targetId,
          route,
          conditional: true,
        }));
    }
    return edge.to === targetId ? [{
      id: `${edgeIndex}:${edge.from}:${targetId}`,
      edgeIndex,
      source: edge.from,
      target: targetId,
      conditional: false,
    }] : [];
  });
}

export function isIsolatedBuilderNode(state: AgentWorkflowBuilderState, nodeId: string): boolean {
  return !state.edges.some((edge) => (
    edge.from === nodeId
    || edge.to === nodeId
    || Object.values(edge.routes || {}).includes(nodeId)
  ));
}

export function wouldCreateBuilderCycle(
  state: AgentWorkflowBuilderState,
  sourceId: string,
  targetId: string,
): boolean {
  const adjacency = new Map<string, Set<string>>();
  state.edges.forEach((edge) => {
    const targets = edge.conditional ? Object.values(edge.routes || {}) : edge.to ? [edge.to] : [];
    targets.forEach((target) => {
      adjacency.set(edge.from, new Set([...(adjacency.get(edge.from) || []), target]));
    });
  });
  const pending = [targetId];
  const seen = new Set<string>();
  while (pending.length) {
    const current = pending.pop()!;
    if (current === sourceId) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    pending.push(...(adjacency.get(current) || []));
  }
  return false;
}

const canConnectTypeToTarget = (
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  sourceType: string,
  targetId: string,
): CompatibilityResult => {
  if (targetId === 'END') {
    return (catalogEntry(catalog, sourceType)?.allowed_child_types || []).includes('END')
      ? { ok: true }
      : { ok: false, reason: `${sourceType} cannot end the graph.` };
  }
  return canConnectNodeTypes(catalog, sourceType, getNode(state, targetId)?.type);
};

const canConnectSourceToType = (
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  sourceId: string,
  targetType: string,
): CompatibilityResult => {
  if (sourceId === 'START') {
    return (catalogEntry(catalog, targetType)?.allowed_parent_types || []).includes('START')
      ? { ok: true }
      : { ok: false, reason: `${targetType} cannot start the graph.` };
  }
  return canConnectNodeTypes(catalog, getNode(state, sourceId)?.type, targetType);
};

export function canInsertNodeTypeBefore(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  targetId: string,
  nodeType: string,
  incomingPath?: BuilderIncomingPath,
): CompatibilityResult {
  const available = canAddNodeType(catalog, state, nodeType);
  if (!available.ok) return available;
  if (getAllowedRouteFunctionsForNode(catalog, nodeType).length > 0) {
    return { ok: false, reason: `${nodeType} needs named outgoing routes and cannot be inserted as a simple previous step.` };
  }
  if (incomingPath) {
    const before = canConnectSourceToType(catalog, state, incomingPath.source, nodeType);
    if (!before.ok) return before;
  }
  return canConnectTypeToTarget(catalog, state, nodeType, targetId);
}

export function canInsertExistingNodeBefore(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  targetId: string,
  nodeId: string,
  incomingPath?: BuilderIncomingPath,
): CompatibilityResult {
  const node = getNode(state, nodeId);
  if (!node) return { ok: false, reason: `Unknown node: ${nodeId}` };
  if (!isIsolatedBuilderNode(state, nodeId)) {
    return { ok: false, reason: 'Only unconnected nodes can be inserted without changing other paths.' };
  }
  if (nodeId === targetId || nodeId === incomingPath?.source) {
    return { ok: false, reason: 'A path cannot be inserted through itself.' };
  }
  if (wouldCreateBuilderCycle(state, nodeId, targetId)) {
    return { ok: false, reason: 'This insertion would create a cycle.' };
  }
  if (getAllowedRouteFunctionsForNode(catalog, node.type).length > 0) {
    return { ok: false, reason: `${node.type} needs named outgoing routes and cannot be inserted as a simple previous step.` };
  }
  if (incomingPath) {
    const before = canConnectNodes(catalog, state, incomingPath.source, nodeId);
    if (!before.ok) return before;
  }
  return canConnectNodes(catalog, state, nodeId, targetId);
}

export function insertNodeBefore(
  state: AgentWorkflowBuilderState,
  targetId: string,
  node: BuilderNodeState,
  incomingPath?: BuilderIncomingPath,
  addNode = true,
): AgentWorkflowBuilderState {
  const edges = state.edges.map((edge, edgeIndex) => {
    if (!incomingPath || edgeIndex !== incomingPath.edgeIndex) return edge;
    if (incomingPath.conditional && incomingPath.route) {
      return {
        ...edge,
        routes: { ...(edge.routes || {}), [incomingPath.route]: node.id },
      };
    }
    return { ...edge, to: node.id };
  });
  const hasTail = edges.some((edge) => !edge.conditional && edge.from === node.id && edge.to === targetId);
  return {
    ...state,
    nodes: addNode ? [...state.nodes, node] : state.nodes,
    edges: hasTail ? edges : [...edges, { from: node.id, to: targetId }],
  };
}

const nodeWithDefaultTools = (
  catalog: AgentWorkflowCatalogResponse,
  id: string,
  type: string,
  preferredToolIds?: string[],
): BuilderNodeState => {
  const allowed = getAllowedToolContractsForNode(catalog, type).map((contract) => contract.id);
  const selected = preferredToolIds
    ? preferredToolIds.filter((toolId) => allowed.includes(toolId))
    : allowed.slice(0, 1);
  return selected.length > 0 ? { id, type, tool_contract_ids: selected } : { id, type };
};

const defaultContextPolicy = (catalog: AgentWorkflowCatalogResponse) => ({
  ...(catalog.defaults?.context_policy || {}),
  evidence_dedupe: true,
  evidence_compression: 'compact',
  final_context_char_limit: 25536,
});

const createLoopPolicy = (nodes: BuilderNodeState[]) => {
  const repeatableIds = nodes
    .filter((node) => REPEATABLE_NODE_TYPES.has(node.type))
    .map((node) => node.id)
    .sort();
  const effectiveTotal = Math.max(nodes.length + repeatableIds.length, nodes.length);
  return {
    max_total_visits: effectiveTotal,
    default_max_node_visits: 1,
    node_visit_limits: Object.fromEntries(repeatableIds.map((nodeId) => [nodeId, 2])),
  };
};

const normalizeLoopPolicy = (
  nodes: BuilderNodeState[],
  policy?: Record<string, any>,
): Record<string, any> | undefined => {
  if (!policy) return undefined;
  const nodeIds = new Set(nodes.map((node) => node.id));
  const parsedDefault = Number(policy.default_max_node_visits);
  const defaultMax = Number.isInteger(parsedDefault) && parsedDefault > 0 ? parsedDefault : 1;
  const nodeVisitLimits = Object.fromEntries(
    Object.entries(
      policy.node_visit_limits && typeof policy.node_visit_limits === 'object'
        ? policy.node_visit_limits
        : {},
    ).flatMap(([nodeId, rawLimit]) => {
      const limit = Number(rawLimit);
      return nodeIds.has(nodeId) && Number.isInteger(limit) && limit > 0
        ? [[nodeId, limit]]
        : [];
    }),
  );
  const effectiveTotal = nodes.reduce(
    (total, node) => total + Number(nodeVisitLimits[node.id] || defaultMax),
    0,
  );
  const parsedTotal = Number(policy.max_total_visits);
  const requestedTotal = Number.isInteger(parsedTotal) && parsedTotal > 0
    ? parsedTotal
    : effectiveTotal;
  return {
    ...policy,
    max_total_visits: Math.max(nodes.length, Math.min(requestedTotal, effectiveTotal)),
    default_max_node_visits: defaultMax,
    node_visit_limits: nodeVisitLimits,
  };
};

const collectAllowedToolIds = (nodes: BuilderNodeState[]) => (
  Array.from(new Set(nodes.flatMap((node) => node.tool_contract_ids || []))).sort()
);

const findPrimaryRouteTarget = (edge?: BuilderEdgeState): string | undefined => {
  if (!edge?.routes) return undefined;
  return edge.routes.approve
    || edge.routes.approve_selected
    || edge.routes.continue_without
    || Object.values(edge.routes).find((target) => target !== 'END')
    || Object.values(edge.routes)[0];
};

const materializeHitlPolicy = (state: AgentWorkflowBuilderState) => {
  const hitlNodes = state.nodes.filter((node) => node.type === BuiltinAgentNodeType.HitlGate && node.hitl);
  if (hitlNodes.length === 0) return state.hitl_policy ? clone(state.hitl_policy) : undefined;
  const base = state.hitl_policy ? clone(state.hitl_policy) : {};
  const gates = { ...(base.gates || {}) };
  hitlNodes.forEach((node) => {
    const routeEdge = state.edges.find((edge) => edge.from === node.id && edge.conditional);
    const targetNodeId = findPrimaryRouteTarget(routeEdge);
    gates[node.id] = {
      enabled: true,
      mode: node.hitl?.mode || HitlMode.Approval,
      phase: node.hitl?.phase || HitlPhase.Before,
      target: targetNodeId && targetNodeId !== 'END' ? { node_id: targetNodeId } : { node_type: 'finalizer' },
      title: node.hitl?.title || `Review ${targetNodeId || node.id}`,
      body: node.hitl?.body || '',
      prompt: node.hitl?.prompt || node.hitl?.body || '',
      allowed_actions: node.hitl?.allowed_actions || [AgentRunResumeAction.Approve, AgentRunResumeAction.Reject, AgentRunResumeAction.ContinueWithout],
      default_action: node.hitl?.default_action || AgentRunResumeAction.ContinueWithout,
      routes: routeEdge?.routes ? clone(routeEdge.routes) : clone(node.hitl?.routes || {}),
    };
  });
  return {
    ...base,
    enabled: true,
    gates,
  };
};

export function createInitialBuilderState(
  catalog: AgentWorkflowCatalogResponse,
  starter: AgentWorkflowStarter = 'router',
): AgentWorkflowBuilderState {
  if (starter === 'plan_execute') {
    const nodes = [
      nodeWithDefaultTools(catalog, 'context_loader', 'context_loader'),
      nodeWithDefaultTools(catalog, 'planner', 'planner'),
      nodeWithDefaultTools(catalog, 'retrieval_worker', 'retrieval_worker', ['document_evidence']),
      { id: 'synthesizer', type: 'synthesizer' },
      nodeWithDefaultTools(catalog, 'finalizer', 'finalizer'),
    ];
    return {
      workflowType: 'custom_rag_agent',
      nodes,
      edges: [
        { from: 'START', to: 'context_loader' },
        { from: 'context_loader', to: 'planner' },
        {
          from: 'planner',
          conditional: true,
          route_fn: getDefaultRouteFunctionForNode(catalog, BuiltinAgentNodeType.Planner) || RouteFunctionId.Planner,
          routes: { execute: 'retrieval_worker', direct: 'finalizer', clarify: 'finalizer' },
        },
        { from: 'retrieval_worker', to: 'synthesizer' },
        { from: 'synthesizer', to: 'finalizer' },
        { from: 'finalizer', to: 'END' },
      ],
      allowed_tool_ids: collectAllowedToolIds(nodes),
      context_policy: defaultContextPolicy(catalog),
      loop_policy: createLoopPolicy(nodes),
      runtime: defaultRuntime(false, 'planner'),
    };
  }

  if (starter === 'evaluator_replanner') {
    const nodes = [
      nodeWithDefaultTools(catalog, 'context_loader', 'context_loader'),
      nodeWithDefaultTools(catalog, 'planner', 'planner'),
      nodeWithDefaultTools(catalog, 'retrieval_worker', 'retrieval_worker', ['document_evidence']),
      nodeWithDefaultTools(catalog, 'evidence_evaluator', 'evidence_evaluator'),
      nodeWithDefaultTools(catalog, 'replanner', 'replanner'),
      { id: 'synthesizer', type: 'synthesizer' },
      nodeWithDefaultTools(catalog, 'finalizer', 'finalizer'),
    ];
    return {
      workflowType: 'custom_rag_agent',
      nodes,
      edges: [
        { from: 'START', to: 'context_loader' },
        { from: 'context_loader', to: 'planner' },
        {
          from: 'planner',
          conditional: true,
          route_fn: getDefaultRouteFunctionForNode(catalog, BuiltinAgentNodeType.Planner) || RouteFunctionId.Planner,
          routes: { execute: 'retrieval_worker', direct: 'finalizer', clarify: 'finalizer' },
        },
        { from: 'retrieval_worker', to: 'evidence_evaluator' },
        {
          from: 'evidence_evaluator',
          conditional: true,
          route_fn: getDefaultRouteFunctionForNode(catalog, BuiltinAgentNodeType.EvidenceEvaluator) || RouteFunctionId.Evaluator,
          routes: { answer: 'synthesizer', replan: 'replanner', answer_budget_exhausted: 'synthesizer' },
        },
        { from: 'replanner', to: 'retrieval_worker' },
        { from: 'synthesizer', to: 'finalizer' },
        { from: 'finalizer', to: 'END' },
      ],
      allowed_tool_ids: collectAllowedToolIds(nodes),
      context_policy: defaultContextPolicy(catalog),
      loop_policy: createLoopPolicy(nodes),
      runtime: defaultRuntime(true, 'evaluator_replanner'),
    };
  }

  const nodes = [
    nodeWithDefaultTools(catalog, 'context_loader', 'context_loader'),
    nodeWithDefaultTools(catalog, 'router', 'router'),
    nodeWithDefaultTools(catalog, 'retrieval_worker', 'retrieval_worker', ['document_evidence']),
    { id: 'synthesizer', type: 'synthesizer' },
    nodeWithDefaultTools(catalog, 'finalizer', 'finalizer'),
  ];
  return {
    workflowType: 'custom_rag_agent',
    nodes,
    edges: [
      { from: 'START', to: 'context_loader' },
      { from: 'context_loader', to: 'router' },
      {
        from: 'router',
        conditional: true,
        route_fn: getDefaultRouteFunctionForNode(catalog, BuiltinAgentNodeType.Router) || RouteFunctionId.Router,
        routes: { document: 'retrieval_worker', direct: 'finalizer', clarify: 'finalizer' },
      },
      { from: 'retrieval_worker', to: 'synthesizer' },
      { from: 'synthesizer', to: 'finalizer' },
      { from: 'finalizer', to: 'END' },
    ],
    allowed_tool_ids: collectAllowedToolIds(nodes),
    context_policy: defaultContextPolicy(catalog),
    loop_policy: createLoopPolicy(nodes),
    runtime: defaultRuntime(false, 'router'),
  };
}

export function assembleAgentWorkflowSpec(
  state: AgentWorkflowBuilderState,
  overrides: Record<string, any> = {},
): AgentWorkflowBuilderSpec {
  const hitlPolicy = materializeHitlPolicy(state);
  const config = {
    ...(state.extraConfig || {}),
    ...(state.context_policy ? { context_policy: clone(state.context_policy) } : {}),
    ...(state.loop_policy ? { loop_policy: clone(state.loop_policy) } : {}),
    ...(hitlPolicy ? { hitl_policy: hitlPolicy } : {}),
    allowed_tool_ids: [...(state.allowed_tool_ids?.length ? state.allowed_tool_ids : collectAllowedToolIds(state.nodes))],
    builder_ui: {
      ...(state.builder_ui || {}),
      positions: Object.fromEntries(state.nodes.filter((node) => node.position).map((node) => [node.id, node.position])),
    },
    graph: {
      nodes: state.nodes.map((node) => {
        const { hitl, position, ...rest } = node;
        return clone(rest);
      }),
      edges: clone(state.edges),
    },
    ...overrides,
  };
  return {
    schema_version: 2,
    workflow_id: state.workflowId || 'custom_rag_agent',
    workflow_type: state.workflowType || 'custom_rag_agent',
    runtime: clone(state.runtime || defaultRuntime(false)),
    config,
  };
}

export function loadBuilderStateFromSpec(spec: AgentWorkflowBuilderSpec | Record<string, any>): AgentWorkflowBuilderState {
  const config = spec?.config && typeof spec.config === 'object' ? spec.config : {};
  const graph = config.graph && typeof config.graph === 'object' ? config.graph as AgentWorkflowGraphSpec : {};
  const nodes = Array.isArray(graph.nodes) ? graph.nodes.map((node) => clone(node) as BuilderNodeState) : [];
  const edges = Array.isArray(graph.edges) ? graph.edges.map((edge) => clone(edge) as BuilderEdgeState) : [];
  const knownConfigKeys = new Set(['graph', 'allowed_tool_ids', 'context_policy', 'loop_policy', 'hitl_policy', 'builder_ui']);
  const extraConfig = Object.fromEntries(
    Object.entries(config).filter(([key]) => !knownConfigKeys.has(key)),
  );
  const builderUi = config.builder_ui && typeof config.builder_ui === 'object' ? clone(config.builder_ui) : {};
  const positions = builderUi.positions && typeof builderUi.positions === 'object' ? builderUi.positions : {};
  return {
    workflowId: typeof spec.workflow_id === 'string' && spec.workflow_id
      ? spec.workflow_id
      : 'custom_rag_agent',
    workflowType: typeof spec.workflow_type === 'string' ? spec.workflow_type : 'custom_rag_agent',
    nodes: nodes.map((node) => positions[node.id] ? { ...node, position: clone(positions[node.id]) } : node),
    edges,
    allowed_tool_ids: Array.isArray(config.allowed_tool_ids) ? [...config.allowed_tool_ids] : collectAllowedToolIds(nodes),
    context_policy: config.context_policy ? clone(config.context_policy) : undefined,
    loop_policy: config.loop_policy ? clone(config.loop_policy) : undefined,
    hitl_policy: config.hitl_policy ? clone(config.hitl_policy) : undefined,
    runtime: spec.runtime && typeof spec.runtime === 'object' ? clone(spec.runtime) : defaultRuntime(false),
    extraConfig,
    builder_ui: {
      notes: Array.isArray(builderUi.notes) ? builderUi.notes : [],
      groups: Array.isArray(builderUi.groups) ? builderUi.groups : [],
    },
  };
}

export function normalizeBuilderState(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
): AgentWorkflowBuilderState {
  const seenTypeCounts = new Map<string, number>();
  const nodes = state.nodes.filter((node) => {
    const entry = catalogEntry(catalog, node.type);
    if (!entry) return false;
    const count = seenTypeCounts.get(node.type) || 0;
    if (count >= entry.max_instances) return false;
    seenTypeCounts.set(node.type, count + 1);
    const allowedTools = new Set(entry.allowed_tool_contract_ids || []);
    const selectedTools = (node.tool_contract_ids || []).filter((toolId) => allowedTools.has(toolId));
    return Object.assign(node, selectedTools.length > 0 ? { tool_contract_ids: selectedTools } : { tool_contract_ids: undefined });
  }).map((node) => {
    const { tool_contract_ids, ...rest } = node;
    return tool_contract_ids?.length ? { ...rest, tool_contract_ids } : rest;
  });
  const normalized: AgentWorkflowBuilderState = {
    ...state,
    nodes,
    loop_policy: normalizeLoopPolicy(nodes, state.loop_policy),
    edges: state.edges.filter((edge) => {
      if (edge.conditional) {
        return Boolean(edge.from && edge.route_fn && edge.routes && Object.keys(edge.routes).length > 0);
      }
      return Boolean(edge.from && edge.to);
    }),
  };
  normalized.allowed_tool_ids = collectAllowedToolIds(normalized.nodes);
  return normalized;
}

export function createHitlGateForTarget(
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  targetNodeId: string,
  options: {
    id?: string;
    sourceNodeId?: string;
    title?: string;
    body?: string;
    mode?: string;
    allowedActions?: string[];
    defaultAction?: string;
  } = {},
): AgentWorkflowBuilderState {
  const target = getNode(state, targetNodeId);
  if (!target) return state;
  const gateId = options.id || getCanonicalNodeId(`hitl_${targetNodeId}`, state.nodes.map((node) => node.id));
  const gate: BuilderNodeState = {
    id: gateId,
    type: BuiltinAgentNodeType.HitlGate,
    hitl: {
      title: options.title || `Review ${target.id}`,
      body: options.body || '',
      mode: options.mode || HitlMode.Approval,
      allowed_actions: options.allowedActions || [AgentRunResumeAction.Approve, AgentRunResumeAction.Reject, AgentRunResumeAction.ContinueWithout],
      default_action: options.defaultAction || AgentRunResumeAction.ContinueWithout,
      routes: {
        [AgentRunResumeAction.Approve]: target.id,
        [AgentRunResumeAction.ContinueWithout]: target.id,
        [AgentRunResumeAction.Reject]: 'END',
      },
    },
  };
  const sourceEdge = options.sourceNodeId
    ? state.edges.find((edge) => edge.from === options.sourceNodeId && edge.to === targetNodeId)
    : state.edges.find((edge) => edge.to === targetNodeId && !edge.conditional);
  const edges = state.edges.map((edge) => (
    sourceEdge && edge === sourceEdge ? { ...edge, to: gateId } : edge
  ));
  if (!sourceEdge) {
    edges.push({ from: 'START', to: gateId });
  }
  edges.push({
    from: gateId,
    conditional: true,
    route_fn: getDefaultRouteFunctionForNode(catalog, BuiltinAgentNodeType.HitlGate) || RouteFunctionId.HitlGate,
    routes: gate.hitl?.routes || {},
  });
  return {
    ...state,
    nodes: [...state.nodes, gate],
    edges,
  };
}
