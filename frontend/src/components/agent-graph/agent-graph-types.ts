import type { AgentGraphMode, AgentGraphNodeStatus } from '../../lib/enums.ts';

export type { AgentGraphMode, AgentGraphNodeStatus };

export interface AgentNodeCatalogEntry {
  displayName?: string;
  display_name?: string;
  category?: string;
  capabilities?: unknown;
  observability?: unknown;
  [key: string]: unknown;
}

export type AgentNodeCatalog = Record<string, AgentNodeCatalogEntry>;

/** Identifies one invocation of a workflow node within a run. */
export interface AgentNodeVisitRef {
  nodeId: string;
  visitIndex: number;
}

export interface AgentWorkflowGraphNodeSpec {
  id: string;
  type: string;
  position?: { x: number; y: number };
  [key: string]: any;
}

export interface AgentWorkflowGraphEdgeSpec {
  from: string;
  to?: string;
  conditional?: boolean;
  routes?: Record<string, string>;
  [key: string]: any;
}

export interface AgentWorkflowGraphSpec {
  nodes?: AgentWorkflowGraphNodeSpec[];
  edges?: AgentWorkflowGraphEdgeSpec[];
}

export interface AgentGraphToolSummary {
  toolName: string;
  displayName?: string;
  callerNode?: string;
  callerNodeType?: string;
  callerVisitIndex?: number;
  ok: boolean;
  elapsedMs?: number;
  sourceCount?: number;
  warnings: string[];
  artifactKeys: string[];
  toolInput?: unknown;
  resultPreview?: string;
  artifactRefs?: Record<string, any>;
  artifactSummary?: Record<string, any>;
  traceSpan?: Record<string, any>;
  raw: Record<string, any>;
}

export interface AgentGraphNodeVisit {
  visitIndex?: number;
  label: string;
  status: AgentGraphNodeStatus;
  elapsedMs?: number;
  route?: string;
  routeReason?: string;
  evaluatorRoute?: string;
  replanCount?: number;
  warningCount: number;
  errorCount: number;
  toolCount: number;
  rawEvents: Record<string, any>[];
  toolSummaries: AgentGraphToolSummary[];
}

export interface AgentTraceRefs {
  node_ids?: string[];
  span_ids?: string[];
  interrupt_id?: string | null;
  [key: string]: unknown;
}

export interface AgentGraphNode {
  [key: string]: unknown;
  id: string;
  type: string;
  label: string;
  category?: string;
  capabilities?: string[];
  observability?: Record<string, unknown>;
  instanceId?: string;
  instanceLabel?: string;
  description?: string;
  position?: { x: number; y: number };
  status: AgentGraphNodeStatus;
  focused?: boolean;
  focusedSpanIds?: string[];
  focusedTraceSpans?: Record<string, any>[];
  elapsedMs?: number;
  route?: string;
  routeReason?: string;
  skipped?: boolean;
  skipReason?: string;
  executionPlan?: string[];
  warnings?: string[];
  visitCount?: number;
  visits?: AgentGraphNodeVisit[];
  latestVisitIndex?: number;
  selectedVisitIndex?: number;
  selectedVisitPosition?: number;
  inputRefs?: Record<string, any>;
  outputRefs?: Record<string, any>;
  inputPreview?: unknown;
  outputPreview?: unknown;
  promptSummary?: Record<string, any>;
  llmResultSummary?: Record<string, any>;
  llmSummary?: Record<string, any>;
  layoutDirection?: 'RIGHT' | 'DOWN';
  toolSummaries: AgentGraphToolSummary[];
  warningCount: number;
  errorCount: number;
  sourceCount: number;
  artifactCount: number;
  traceSpans?: Record<string, any>[];
  rawEvents: Record<string, any>[];
  authoring?: boolean;
  inputLabel?: string;
  outputPorts?: { id: string; label: string; description?: string }[];
  compatible?: boolean;
  compatibilityReason?: string;
  issueCount?: number;
  usesLlm?: boolean;
  usesTools?: boolean;
  onAddNext?: (nodeId: string, route?: string) => void;
  onAddPrevious?: (nodeId: string) => void;
}

export interface AgentGraphEdge {
  [key: string]: unknown;
  id: string;
  source: string;
  target: string;
  label?: string;
  route?: string;
  selected: boolean;
  active: boolean;
  conditional: boolean;
  raw?: Record<string, any>;
}

export interface AgentGraphRuntimeOverlay {
  route?: string;
  routeReason?: string;
  executionPlan?: string[];
  nodeRows?: Record<string, any>[];
  toolRows?: Record<string, any>[];
  errors?: Record<string, any>[];
  metrics?: Record<string, any>;
  nodeCatalog?: AgentNodeCatalog;
}

export type AgentGraphSelection =
  | { kind: 'node'; node: AgentGraphNode }
  | { kind: 'edge'; edge: AgentGraphEdge }
  | null;
