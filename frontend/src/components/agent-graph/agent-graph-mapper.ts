import type {
  AgentGraphEdge,
  AgentGraphNode,
  AgentGraphNodeStatus,
  AgentGraphRuntimeOverlay,
  AgentGraphToolSummary,
  AgentTraceRefs,
  AgentPatternGraphSpec,
} from './agent-graph-types';
export { formatNodeLabel, formatNodeInstanceLabel } from './agent-node-labels.js';
import { formatNodeLabel, formatNodeInstanceLabel } from './agent-node-labels.js';

const BUILTIN_GRAPHS: Record<string, AgentPatternGraphSpec> = {
  router_rag_agent: {
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
  plan_execute_rag_agent: {
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
        routes: {
          execute: 'retrieval_worker',
          direct: 'direct_answer',
          clarify: 'finalizer',
        },
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
  evaluator_replanner_rag_agent: {
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
        routes: {
          execute: 'retrieval_worker',
          direct: 'direct_answer',
          clarify: 'finalizer',
        },
      },
      { from: 'direct_answer', to: 'finalizer' },
      { from: 'retrieval_worker', to: 'memory_worker' },
      { from: 'memory_worker', to: 'timeline_worker' },
      { from: 'timeline_worker', to: 'web_worker' },
      { from: 'web_worker', to: 'evidence_evaluator' },
      {
        from: 'evidence_evaluator',
        conditional: true,
        routes: {
          answer: 'synthesizer',
          replan: 'replanner',
          answer_budget_exhausted: 'synthesizer',
        },
      },
      { from: 'replanner', to: 'retrieval_worker' },
      { from: 'synthesizer', to: 'finalizer' },
      { from: 'finalizer', to: 'END' },
    ],
  },
};

const asArray = (value: any): Record<string, any>[] => (
  Array.isArray(value) ? value.filter((item): item is Record<string, any> => item && typeof item === 'object') : []
);

const asObject = (value: any): Record<string, any> | undefined => (
  value && typeof value === 'object' && !Array.isArray(value) ? value : undefined
);

const asStringArray = (value: any): string[] | undefined => {
  if (!Array.isArray(value)) return undefined;
  const items = value.filter((item): item is string => typeof item === 'string' && item.length > 0);
  return items.length > 0 ? items : undefined;
};

const unique = (items: string[]) => Array.from(new Set(items.filter(Boolean)));

const getNodeIdFromEvent = (event: Record<string, any>) => (
  typeof event.node === 'string' ? event.node : typeof event.name === 'string' ? event.name : ''
);

const getExecutionPlan = (overlay: AgentGraphRuntimeOverlay, nodeRows: Record<string, any>[]) => {
  if (Array.isArray(overlay.executionPlan)) return overlay.executionPlan.filter((item): item is string => typeof item === 'string');
  for (let index = nodeRows.length - 1; index >= 0; index -= 1) {
    const plan = nodeRows[index]?.execution_plan;
    if (Array.isArray(plan)) return plan.filter((item): item is string => typeof item === 'string');
  }
  return [];
};

const summarizeTool = (event: Record<string, any>): AgentGraphToolSummary => ({
  toolName: String(event.tool_name || 'tool'),
  displayName: typeof event.tool_display_name === 'string' ? event.tool_display_name : undefined,
  callerNode: typeof event.caller_node === 'string' ? event.caller_node : undefined,
  callerNodeType: typeof event.caller_node_type === 'string' ? event.caller_node_type : undefined,
  ok: event.ok !== false,
  elapsedMs: Number.isFinite(Number(event.elapsed_ms)) ? Number(event.elapsed_ms) : undefined,
  sourceCount: Number.isFinite(Number(event.source_count)) ? Number(event.source_count) : undefined,
  warnings: Array.isArray(event.warnings) ? event.warnings.map(String) : [],
  artifactKeys: Array.isArray(event.artifact_keys) ? event.artifact_keys.map(String) : [],
  toolInput: event.tool_input,
  resultPreview: typeof event.result_preview === 'string' ? event.result_preview : undefined,
  artifactRefs: event.artifact_refs && typeof event.artifact_refs === 'object' ? event.artifact_refs : undefined,
  artifactSummary: event.artifact_summary && typeof event.artifact_summary === 'object' ? event.artifact_summary : undefined,
  traceSpan: event.__trace_span && typeof event.__trace_span === 'object' ? event.__trace_span : undefined,
  raw: event,
});

const deriveStatus = (
  nodeId: string,
  events: Record<string, any>[],
  toolSummaries: AgentGraphToolSummary[],
  executionPlan: string[],
): AgentGraphNodeStatus => {
  if (events.some((event) => event.error || event.ok === false) || toolSummaries.some((tool) => !tool.ok)) return 'error';
  if (events.some((event) => event.skipped === true)) return 'skipped';
  if (events.length > 0) return 'active';
  if (executionPlan.includes(nodeId)) return 'planned';
  return 'inactive';
};

const hasActiveNode = (nodeId: string, nodesById: Map<string, AgentGraphNode>) => {
  const status = nodesById.get(nodeId)?.status;
  return status === 'active' || status === 'planned' || status === 'skipped' || status === 'error';
};

const selectedConditionalRoute = (
  source: string,
  route: string,
  selectedRoute: any,
  eventsByNode: Map<string, Record<string, any>[]>,
) => {
  if (selectedRoute === route) return true;
  const sourceEvents = eventsByNode.get(source) || [];
  if (source === 'evidence_evaluator') {
    return sourceEvents.some((event) => event.evaluator_route === route || event.evaluatorRoute === route);
  }
  return false;
};

export const getAgentGraphSpec = (resolvedSpec?: Record<string, any>, templateId?: string): AgentPatternGraphSpec => {
  const graph = resolvedSpec?.config?.graph;
  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) return graph;
  return BUILTIN_GRAPHS[templateId || resolvedSpec?.pattern_type || ''] || { nodes: [], edges: [] };
};

export const buildAgentGraph = (
  graphSpec: AgentPatternGraphSpec,
  overlay: AgentGraphRuntimeOverlay = {},
) => {
  const nodeRows = asArray(overlay.nodeRows);
  const toolRows = asArray(overlay.toolRows);
  const nodeCatalog = overlay.nodeCatalog;
  const executionPlan = getExecutionPlan(overlay, nodeRows);
  const selectedRoute = overlay.route || overlay.metrics?.route;

  const eventsByNode = new Map<string, Record<string, any>[]>();
  nodeRows.forEach((event) => {
    const nodeId = getNodeIdFromEvent(event);
    if (!nodeId) return;
    eventsByNode.set(nodeId, [...(eventsByNode.get(nodeId) || []), event]);
  });

  const toolsByNode = new Map<string, AgentGraphToolSummary[]>();
  toolRows.forEach((event) => {
    const tool = summarizeTool(event);
    if (!tool.callerNode) return;
    toolsByNode.set(tool.callerNode, [...(toolsByNode.get(tool.callerNode) || []), tool]);
  });

  const nodes: AgentGraphNode[] = (graphSpec.nodes || [])
    .filter((node) => node && typeof node.id === 'string' && typeof node.type === 'string')
    .map((node) => {
      const rawEvents = eventsByNode.get(node.id) || [];
      const toolSummaries = toolsByNode.get(node.id) || [];
      const elapsedMs = rawEvents.reduce((total, event) => total + (Number(event.elapsed_ms) || 0), 0);
      const latestEvent = rawEvents[rawEvents.length - 1] || {};
      const status = deriveStatus(node.id, rawEvents, toolSummaries, executionPlan);
      const traceSpans = rawEvents
        .map((event) => event.__trace_span)
        .filter((span): span is Record<string, any> => span && typeof span === 'object');
      const catalogEntry = nodeCatalog?.[node.type];
      const category = typeof node.category === 'string'
        ? node.category
        : typeof catalogEntry?.category === 'string'
          ? catalogEntry.category
          : undefined;
      return {
        id: node.id,
        type: node.type,
        label: formatNodeLabel(node.id, node.type, nodeCatalog),
        category,
        capabilities: asStringArray(node.capabilities) || asStringArray(catalogEntry?.capabilities),
        observability: asObject(node.observability) || asObject(catalogEntry?.observability),
        instanceId: node.id,
        instanceLabel: formatNodeInstanceLabel(node.id, node.type),
        description: typeof node.description === 'string' ? node.description : undefined,
        status,
        elapsedMs: elapsedMs > 0 ? elapsedMs : undefined,
        route: typeof latestEvent.route === 'string' ? latestEvent.route : undefined,
        routeReason: typeof latestEvent.route_reason === 'string' ? latestEvent.route_reason : undefined,
        skipped: latestEvent.skipped === true,
        skipReason: typeof latestEvent.skip_reason === 'string' ? latestEvent.skip_reason : undefined,
        executionPlan: node.id === 'planner' || node.id === 'replanner' ? executionPlan : undefined,
        warnings: rawEvents.flatMap((event) => (Array.isArray(event.warnings) ? event.warnings.map(String) : [])),
        inputRefs: latestEvent.input_refs && typeof latestEvent.input_refs === 'object' ? latestEvent.input_refs : undefined,
        outputRefs: latestEvent.output_refs && typeof latestEvent.output_refs === 'object' ? latestEvent.output_refs : undefined,
        inputPreview: latestEvent.input_preview,
        outputPreview: latestEvent.output_preview,
        promptSummary: latestEvent.prompt_summary && typeof latestEvent.prompt_summary === 'object' ? latestEvent.prompt_summary : undefined,
        llmResultSummary: latestEvent.llm_result_summary && typeof latestEvent.llm_result_summary === 'object' ? latestEvent.llm_result_summary : undefined,
        llmSummary: latestEvent.llm_summary && typeof latestEvent.llm_summary === 'object' ? latestEvent.llm_summary : undefined,
        toolSummaries,
        warningCount: toolSummaries.reduce((count, tool) => count + tool.warnings.length, 0)
          + rawEvents.reduce((count, event) => count + (Array.isArray(event.warnings) ? event.warnings.length : 0), 0),
        errorCount: toolSummaries.filter((tool) => !tool.ok).length + rawEvents.filter((event) => event.error || event.ok === false).length,
        sourceCount: toolSummaries.reduce((count, tool) => count + (tool.sourceCount || 0), 0),
        artifactCount: toolSummaries.reduce((count, tool) => count + tool.artifactKeys.length, 0),
        traceSpans: traceSpans.length > 0 ? traceSpans : undefined,
        rawEvents,
      };
    });

  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const edges: AgentGraphEdge[] = [];
  (graphSpec.edges || []).forEach((edge, edgeIndex) => {
    if (!edge || typeof edge.from !== 'string') return;
    if (edge.from === 'START' || edge.to === 'END') return;

    if (edge.conditional && edge.routes && typeof edge.routes === 'object') {
      Object.entries(edge.routes).forEach(([route, target]) => {
        if (typeof target !== 'string') return;
        const selected = selectedConditionalRoute(edge.from, route, selectedRoute, eventsByNode);
        edges.push({
          id: `${edge.from}-${route}-${target}`,
          source: edge.from,
          target,
          label: route,
          route,
          selected,
          active: selected || (hasActiveNode(edge.from, nodesById) && hasActiveNode(target, nodesById)),
          conditional: true,
          raw: edge,
        });
      });
      return;
    }

    if (typeof edge.to !== 'string') return;
    edges.push({
      id: `${edge.from}-${edge.to}-${edgeIndex}`,
      source: edge.from,
      target: edge.to,
      selected: false,
      active: hasActiveNode(edge.from, nodesById) && hasActiveNode(edge.to, nodesById),
      conditional: false,
      raw: edge,
    });
  });

  const plannedNodes = unique(executionPlan);
  return { nodes, edges, executionPlan: plannedNodes, selectedRoute };
};

const normalizeRefSet = (value: unknown) => (
  new Set(Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.length > 0) : [])
);

const spanIdFor = (span: unknown) => (
  span && typeof span === 'object' && typeof (span as Record<string, unknown>).span_id === 'string'
    ? String((span as Record<string, unknown>).span_id)
    : undefined
);

const collectNodeSpans = (node: AgentGraphNode) => {
  const nodeSpans = Array.isArray(node.traceSpans) ? node.traceSpans : [];
  const toolSpans = node.toolSummaries
    .map((tool) => tool.traceSpan)
    .filter((span): span is Record<string, any> => Boolean(span));
  return [...nodeSpans, ...toolSpans];
};

export const applyTraceFocusToGraph = (
  graph: ReturnType<typeof buildAgentGraph>,
  refs?: AgentTraceRefs | null,
): ReturnType<typeof buildAgentGraph> => {
  const focusedNodeIds = normalizeRefSet(refs?.node_ids);
  const focusedSpanIds = normalizeRefSet(refs?.span_ids);
  if (focusedNodeIds.size === 0 && focusedSpanIds.size === 0) return graph;

  return {
    ...graph,
    nodes: graph.nodes.map((node) => {
      const spans = collectNodeSpans(node);
      const matchingSpans = spans.filter((span) => {
        const spanId = spanIdFor(span);
        return spanId ? focusedSpanIds.has(spanId) : false;
      });
      const focused = focusedNodeIds.has(node.id) || matchingSpans.length > 0;
      if (!focused) return node;
      return {
        ...node,
        focused: true,
        focusedSpanIds: unique([
          ...(node.focusedSpanIds || []),
          ...matchingSpans.map(spanIdFor).filter((spanId): spanId is string => Boolean(spanId)),
        ]),
        focusedTraceSpans: matchingSpans,
      };
    }),
  };
};
