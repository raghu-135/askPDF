export const ProcessStatus = {
  Pending: 'pending',
  Running: 'running',
  Completed: 'completed',
  Failed: 'failed',
  Unknown: 'unknown',
} as const;
export type ProcessStatus = typeof ProcessStatus[keyof typeof ProcessStatus];

export const MessageRole = {
  User: 'user',
  Assistant: 'assistant',
} as const;
export type MessageRole = typeof MessageRole[keyof typeof MessageRole];

export const ThreadFileSourceType = {
  Pdf: 'pdf',
  Browser: 'browser',
} as const;
export type ThreadFileSourceType = typeof ThreadFileSourceType[keyof typeof ThreadFileSourceType];

export const FileStatusSection = {
  Parsing: 'parsing',
  Indexing: 'indexing',
} as const;
export type FileStatusSection = typeof FileStatusSection[keyof typeof FileStatusSection];

export const ReasoningFormat = {
  Structured: 'structured',
  TaggedText: 'tagged_text',
  None: 'none',
} as const;
export type ReasoningFormat = typeof ReasoningFormat[keyof typeof ReasoningFormat];

export const EmbeddingReadinessStatus = {
  Ready: 'ready',
  NotReady: 'not_ready',
  Blocked: 'blocked',
} as const;
export type EmbeddingReadinessStatus = typeof EmbeddingReadinessStatus[keyof typeof EmbeddingReadinessStatus];

export const AgentRunResumeAction = {
  Approve: 'approve',
  ApproveSelected: 'approve_selected',
  Reject: 'reject',
  Edit: 'edit',
  ContinueWithout: 'continue_without',
} as const;
export type AgentRunResumeAction = typeof AgentRunResumeAction[keyof typeof AgentRunResumeAction];

export const AgentRunStatus = {
  Running: 'running',
  AwaitingHuman: 'awaiting_human',
  Completed: 'completed',
  Clarification: 'clarification',
  Failed: 'failed',
  Rejected: 'rejected',
  Expired: 'expired',
} as const;
export type AgentRunStatus = typeof AgentRunStatus[keyof typeof AgentRunStatus];

export const InterruptStatus = {
  Pending: 'pending',
  Resumed: 'resumed',
  Rejected: 'rejected',
  Expired: 'expired',
} as const;
export type InterruptStatus = typeof InterruptStatus[keyof typeof InterruptStatus];

export const HitlMode = {
  Approval: 'approval',
  Choice: 'choice',
  Review: 'review',
} as const;
export type HitlMode = typeof HitlMode[keyof typeof HitlMode];

export const HitlPhase = {
  Before: 'before',
  After: 'after',
  InsideTool: 'inside_tool',
} as const;
export type HitlPhase = typeof HitlPhase[keyof typeof HitlPhase];

export const HitlSelectionMode = {
  Single: 'single',
  Multi: 'multi',
  SingleOrMulti: 'single_or_multi',
} as const;
export type HitlSelectionMode = typeof HitlSelectionMode[keyof typeof HitlSelectionMode];

export const ChatComposerIndexingStatus = {
  Checking: 'checking',
  Indexing: 'indexing',
  Ready: 'ready',
  Blocked: 'blocked',
  Error: 'error',
} as const;
export type ChatComposerIndexingStatus = typeof ChatComposerIndexingStatus[keyof typeof ChatComposerIndexingStatus];

export const ChatComposerStatus = {
  Sending: 'sending',
  NoLlmSelected: 'no_llm_selected',
  LlmChecking: 'llm_checking',
  LlmUnavailable: 'llm_unavailable',
  LlmToolsUnsupported: 'llm_tools_unsupported',
  EmbeddingChecking: 'embedding_checking',
  EmbeddingUnavailable: 'embedding_unavailable',
  IndexError: 'index_error',
  Indexing: 'indexing',
  Ready: 'ready',
} as const;
export type ChatComposerStatus = typeof ChatComposerStatus[keyof typeof ChatComposerStatus];

export const AgentGraphMode = {
  RunDebug: 'run-debug',
  Builder: 'builder',
} as const;
export type AgentGraphMode = typeof AgentGraphMode[keyof typeof AgentGraphMode];

export const AgentGraphNodeStatus = {
  Active: 'active',
  Planned: 'planned',
  Skipped: 'skipped',
  Inactive: 'inactive',
  Error: 'error',
} as const;
export type AgentGraphNodeStatus = typeof AgentGraphNodeStatus[keyof typeof AgentGraphNodeStatus];

export const RouteFunctionId = {
  Router: 'router_route',
  Planner: 'planner_route',
  Evaluator: 'evaluator_route',
  HitlGate: 'hitl_gate_route',
} as const;
export type RouteFunctionId = typeof RouteFunctionId[keyof typeof RouteFunctionId];

export const BuiltinAgentNodeType = {
  Router: 'router',
  Planner: 'planner',
  EvidenceEvaluator: 'evidence_evaluator',
  HitlGate: 'hitl_gate',
} as const;
export type BuiltinAgentNodeType = typeof BuiltinAgentNodeType[keyof typeof BuiltinAgentNodeType];
