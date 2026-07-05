export type AgentGraphMode = 'run-debug' | 'builder';

export type AgentGraphNodeStatus = 'active' | 'planned' | 'skipped' | 'inactive' | 'error';

export interface AgentPatternGraphNodeSpec {
  id: string;
  type: string;
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

export interface AgentGraphNode {
  [key: string]: unknown;
  id: string;
  type: string;
  label: string;
  description?: string;
  status: AgentGraphNodeStatus;
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
  graphNodeRows?: Record<string, any>[];
  graphToolRows?: Record<string, any>[];
  nodeEvents?: Record<string, any>[];
  toolEvents?: Record<string, any>[];
  errors?: Record<string, any>[];
  metrics?: Record<string, any>;
}

export type AgentGraphSelection =
  | { kind: 'node'; node: AgentGraphNode }
  | { kind: 'edge'; edge: AgentGraphEdge }
  | null;
