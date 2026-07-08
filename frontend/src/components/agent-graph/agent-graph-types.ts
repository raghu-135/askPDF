export type AgentGraphMode = 'run-debug' | 'builder';

export interface AgentNodeCatalogEntry {
  displayName?: string;
  display_name?: string;
  category?: string;
  capabilities?: unknown;
  observability?: unknown;
  [key: string]: unknown;
}

export type AgentNodeCatalog = Record<string, AgentNodeCatalogEntry>;

export type AgentGraphNodeStatus = 'active' | 'planned' | 'skipped' | 'inactive' | 'error';

export interface AgentPatternGraphNodeSpec {
  id: string;
  type: string;
  position?: { x: number; y: number };
  [key: string]: any;
}

export interface AgentPatternGraphEdgeSpec {
  from: string;
  to?: string;
  conditional?: boolean;
  routes?: Record<string, string>;
  [key: string]: any;
}

export interface AgentPatternGraphSpec {
  nodes?: AgentPatternGraphNodeSpec[];
  edges?: AgentPatternGraphEdgeSpec[];
}

export interface AgentGraphToolSummary {
  toolName: string;
  displayName?: string;
  callerNode?: string;
  callerNodeType?: string;
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
