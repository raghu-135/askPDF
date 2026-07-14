import type {
  AgentWorkflowBuilderSpec,
  AgentWorkflowCatalogResponse,
  AgentWorkflowGraphSpec,
  AgentWorkflowNodeCatalogEntry,
  AgentWorkflowRouteFunctionMetadata,
  AgentWorkflowToolContract,
} from './api';

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
  workflowType: string;
  nodes: BuilderNodeState[];
  edges: BuilderEdgeState[];
  allowed_tool_ids: string[];
  context_policy?: Record<string, any>;
  loop_policy?: Record<string, any>;
  hitl_policy?: Record<string, any>;
  runtime?: Record<string, any>;
  extraConfig?: Record<string, any>;
}

export interface CompatibilityResult {
  ok: boolean;
  reason?: string;
}

const ROUTE_FUNCTION_BY_NODE_TYPE: Record<string, string> = {
  router: 'router_route',
  planner: 'planner_route',
  evidence_evaluator: 'evaluator_route',
  hitl_gate: 'hitl_gate_route',
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

const createLoopPolicy = (nodes: BuilderNodeState[], maxTotalVisits?: number) => {
  const repeatableIds = nodes
    .filter((node) => REPEATABLE_NODE_TYPES.has(node.type))
    .map((node) => node.id)
    .sort();
  return {
    max_total_visits: maxTotalVisits || Math.max(nodes.length + repeatableIds.length, nodes.length),
    default_max_node_visits: 1,
    node_visit_limits: Object.fromEntries(repeatableIds.map((nodeId) => [nodeId, 2])),
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
  const hitlNodes = state.nodes.filter((node) => node.type === 'hitl_gate' && node.hitl);
  if (hitlNodes.length === 0) return state.hitl_policy ? clone(state.hitl_policy) : undefined;
  const base = state.hitl_policy ? clone(state.hitl_policy) : {};
  const gates = { ...(base.gates || {}) };
  hitlNodes.forEach((node) => {
    const routeEdge = state.edges.find((edge) => edge.from === node.id && edge.conditional);
    const targetNodeId = findPrimaryRouteTarget(routeEdge);
    gates[node.id] = {
      enabled: true,
      mode: node.hitl?.mode || 'approval',
      phase: node.hitl?.phase || 'before',
      target: targetNodeId && targetNodeId !== 'END' ? { node_id: targetNodeId } : { node_type: 'finalizer' },
      title: node.hitl?.title || `Review ${targetNodeId || node.id}`,
      body: node.hitl?.body || '',
      prompt: node.hitl?.prompt || node.hitl?.body || '',
      allowed_actions: node.hitl?.allowed_actions || ['approve', 'reject', 'continue_without'],
      default_action: node.hitl?.default_action || 'continue_without',
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
          route_fn: getDefaultRouteFunctionForNode(catalog, 'planner') || 'planner_route',
          routes: { execute: 'retrieval_worker', direct: 'finalizer', clarify: 'finalizer' },
        },
        { from: 'retrieval_worker', to: 'synthesizer' },
        { from: 'synthesizer', to: 'finalizer' },
        { from: 'finalizer', to: 'END' },
      ],
      allowed_tool_ids: collectAllowedToolIds(nodes),
      context_policy: defaultContextPolicy(catalog),
      loop_policy: createLoopPolicy(nodes, 9),
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
          route_fn: getDefaultRouteFunctionForNode(catalog, 'planner') || 'planner_route',
          routes: { execute: 'retrieval_worker', direct: 'finalizer', clarify: 'finalizer' },
        },
        { from: 'retrieval_worker', to: 'evidence_evaluator' },
        {
          from: 'evidence_evaluator',
          conditional: true,
          route_fn: getDefaultRouteFunctionForNode(catalog, 'evidence_evaluator') || 'evaluator_route',
          routes: { answer: 'synthesizer', replan: 'replanner', answer_budget_exhausted: 'synthesizer' },
        },
        { from: 'replanner', to: 'retrieval_worker' },
        { from: 'synthesizer', to: 'finalizer' },
        { from: 'finalizer', to: 'END' },
      ],
      allowed_tool_ids: collectAllowedToolIds(nodes),
      context_policy: defaultContextPolicy(catalog),
      loop_policy: createLoopPolicy(nodes, 16),
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
        route_fn: getDefaultRouteFunctionForNode(catalog, 'router') || 'router_route',
        routes: { document: 'retrieval_worker', direct: 'finalizer', clarify: 'finalizer' },
      },
      { from: 'retrieval_worker', to: 'synthesizer' },
      { from: 'synthesizer', to: 'finalizer' },
      { from: 'finalizer', to: 'END' },
    ],
    allowed_tool_ids: collectAllowedToolIds(nodes),
    context_policy: defaultContextPolicy(catalog),
    loop_policy: createLoopPolicy(nodes, 9),
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
    graph: {
      nodes: state.nodes.map((node) => {
        const { hitl, ...rest } = node;
        return clone(rest);
      }),
      edges: clone(state.edges),
    },
    ...overrides,
  };
  return {
    schema_version: 2,
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
  const knownConfigKeys = new Set(['graph', 'allowed_tool_ids', 'context_policy', 'loop_policy', 'hitl_policy']);
  const extraConfig = Object.fromEntries(
    Object.entries(config).filter(([key]) => !knownConfigKeys.has(key)),
  );
  return {
    workflowType: typeof spec.workflow_type === 'string' ? spec.workflow_type : 'custom_rag_agent',
    nodes,
    edges,
    allowed_tool_ids: Array.isArray(config.allowed_tool_ids) ? [...config.allowed_tool_ids] : collectAllowedToolIds(nodes),
    context_policy: config.context_policy ? clone(config.context_policy) : undefined,
    loop_policy: config.loop_policy ? clone(config.loop_policy) : undefined,
    hitl_policy: config.hitl_policy ? clone(config.hitl_policy) : undefined,
    runtime: spec.runtime && typeof spec.runtime === 'object' ? clone(spec.runtime) : defaultRuntime(false),
    extraConfig,
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
    type: 'hitl_gate',
    hitl: {
      title: options.title || `Review ${target.id}`,
      body: options.body || '',
      mode: options.mode || 'approval',
      allowed_actions: options.allowedActions || ['approve', 'reject', 'continue_without'],
      default_action: options.defaultAction || 'continue_without',
      routes: {
        approve: target.id,
        continue_without: target.id,
        reject: 'END',
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
    route_fn: getDefaultRouteFunctionForNode(catalog, 'hitl_gate') || 'hitl_gate_route',
    routes: gate.hitl?.routes || {},
  });
  return {
    ...state,
    nodes: [...state.nodes, gate],
    edges,
  };
}
