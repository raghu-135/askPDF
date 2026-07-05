import type {
  AgentGraphEdge,
  AgentGraphNode,
  AgentGraphNodeStatus,
  AgentGraphRuntimeOverlay,
  AgentGraphToolSummary,
  AgentPatternGraphSpec,
} from './agent-graph-types';

const NODE_LABELS: Record<string, string> = {
  context_loader: 'Context Loader',
  router: 'Router',
  planner: 'Planner',
  retrieval_worker: 'Document Retrieval',
  memory_worker: 'Memory Retrieval',
  timeline_worker: 'Timeline Retrieval',
  web_worker: 'Web Retrieval',
  direct_answer: 'Direct Answer',
  synthesizer: 'Synthesizer',
  finalizer: 'Finalizer',
};

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
};

const asArray = (value: any): Record<string, any>[] => (
  Array.isArray(value) ? value.filter((item): item is Record<string, any> => item && typeof item === 'object') : []
);

const unique = (items: string[]) => Array.from(new Set(items.filter(Boolean)));

const formatNodeLabel = (id: string, type?: string) => (
  NODE_LABELS[id] || NODE_LABELS[type || ''] || id.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
);

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
      return {
        id: node.id,
        type: node.type,
        label: formatNodeLabel(node.id, node.type),
        description: typeof node.description === 'string' ? node.description : undefined,
        status,
        elapsedMs: elapsedMs > 0 ? elapsedMs : undefined,
        route: typeof latestEvent.route === 'string' ? latestEvent.route : undefined,
        routeReason: typeof latestEvent.route_reason === 'string' ? latestEvent.route_reason : undefined,
        skipped: latestEvent.skipped === true,
        skipReason: typeof latestEvent.skip_reason === 'string' ? latestEvent.skip_reason : undefined,
        executionPlan: node.id === 'planner' ? executionPlan : undefined,
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
        const selected = selectedRoute === route;
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
