import {
  deserializeAnnotationItems,
  serializeAnnotationItems,
  type AnnotationTransferItem,
} from "./annotation-utils";
import { getBrowserRuntimeContext } from "./date-utils";
import { API_BASE } from "./api-config";
import { consumeAgentExecutionStream, type AgentExecutionStreamEnvelope } from "./agent-execution-stream";
import {
  ProcessStatus as ProcessStatusEnum,
  ThreadFileSourceType,
  type AgentRunResumeAction as AgentRunResumeActionValue,
  type EmbeddingReadinessStatus as EmbeddingReadinessStatusValue,
  type FileStatusSection as FileStatusSectionValue,
  type HitlMode as HitlModeValue,
  type HitlPhase as HitlPhaseValue,
  type HitlSelectionMode as HitlSelectionModeValue,
  type InterruptStatus as InterruptStatusValue,
  type MessageRole as MessageRoleValue,
  type ProcessStatus as ProcessStatusValue,
  type ReasoningFormat as ReasoningFormatValue,
  type ThreadFileSourceType as ThreadFileSourceTypeValue,
} from "./enums";

// Unified API base - RAG service handles all endpoints.
export { API_BASE } from "./api-config";

// ============ PDF Upload ============

export type ProcessStatus = ProcessStatusValue;

export interface ProcessSection {
  status: ProcessStatus;
  started_at?: string;
  finished_at?: string;
  error?: string;
}

export interface IndexingSection extends ProcessSection {
  chunk_count?: number;
  total_chars?: number;
  reused_existing_embeddings?: boolean;
}

export interface FileIndexingStatus {
  summary: IndexingSection;
  models: Record<string, IndexingSection & {
    threads?: Record<string, IndexingSection>;
  }>;
}

export interface FileStatus {
  file_hash?: string;
  parsing: ProcessSection;
  indexing: IndexingSection;
  indexing_status: FileIndexingStatus;
  updated_at: string;
}

// Helper functions for status checks
export const ProcessStatusHelper = {
  isCompleted: (status: ProcessStatusValue) => status === ProcessStatusEnum.Completed,
  isFailed: (status: ProcessStatusValue) => status === ProcessStatusEnum.Failed,
  isRunning: (status: ProcessStatusValue) => status === ProcessStatusEnum.Running,
  isPending: (status: ProcessStatusValue) => status === ProcessStatusEnum.Pending,
  isTerminal: (status: ProcessStatusValue) => status === ProcessStatusEnum.Completed || status === ProcessStatusEnum.Failed,
};

export interface UploadResponse {
  sentences: any[];
  downloadUrl: string;
  fileHash: string;
  fileName: string;
}

interface RawUploadResponse {
  sentences: any[];
  download_url: string;
  file_hash: string;
  file_name: string;
}

const mapUploadResponse = (raw: RawUploadResponse): UploadResponse => ({
  sentences: raw.sentences,
  downloadUrl: raw.download_url,
  fileHash: raw.file_hash,
  fileName: raw.file_name,
});

export type KnowledgeTarget = { scope: "thread" | "project"; id: string };

const targetPath = (target: KnowledgeTarget) => `${target.scope}s/${target.id}`;

async function responseError(res: Response): Promise<Error & { status?: number }> {
  const message = await res.text();
  const error = new Error(message) as Error & { status?: number };
  error.status = res.status;
  return error;
}

export async function uploadPdfToTarget(file: File, target: KnowledgeTarget): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/${targetPath(target)}/files/upload`, { method: "POST", body: form });
  if (!res.ok) throw await responseError(res);
  return mapUploadResponse(await res.json());
}

export async function uploadPdf(file: File, threadId: string): Promise<UploadResponse> {
  return uploadPdfToTarget(file, { scope: "thread", id: threadId });
}

export async function getTargetFileStatus(
  fileHash: string,
  target: KnowledgeTarget,
  options?: { section?: FileStatusSectionValue },
): Promise<FileStatus | { parsing: ProcessSection } | { indexing: IndexingSection }> {
  const params = new URLSearchParams();
  if (options?.section) params.set("section", options.section);
  const query = params.toString();
  const res = await fetch(`${API_BASE}/api/${targetPath(target)}/files/${fileHash}/status${query ? `?${query}` : ""}`);
  if (!res.ok) throw await responseError(res);
  return res.json();
}

export async function getFileStatus(
  fileHash: string,
  threadId: string,
  options?: {
    section?: FileStatusSectionValue;
  }
): Promise<FileStatus | { parsing: ProcessSection } | { indexing: IndexingSection }> {
  const params = new URLSearchParams();
  if (options?.section) params.set("section", options.section);
  const query = params.toString();
  const url = `${API_BASE}/api/threads/${threadId}/files/${fileHash}/status${query ? `?${query}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw await responseError(res);
  return res.json();
}

export interface PdfData {
  sentences: any[];
  downloadUrl: string;
  fileHash: string;
}

interface RawPdfData {
  sentences: any[];
  download_url: string;
  file_hash: string;
}

const mapPdfData = (raw: RawPdfData): PdfData => ({
  sentences: raw.sentences,
  downloadUrl: raw.download_url,
  fileHash: raw.file_hash,
});

export async function getPdfByHash(fileHash: string, threadId: string): Promise<PdfData> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/${fileHash}`);
  if (!res.ok) throw new Error(await res.text());
  return mapPdfData(await res.json());
}

export async function getParsedSentences(fileHash: string, threadId: string): Promise<{ version: string; sentences: any[] }> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/${fileHash}/sentences`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ============ Thread API ============

export interface Thread {
  id: string;
  name: string;
  project_id?: string | null;
  embeddingModel: string;
  settings?: ThreadSettings;
  thread_metadata?: ThreadMetadata;
  documents_meta?: Record<string, ThreadDocumentMeta>;
  created_at: string;
  message_count?: number;
  file_count?: number;
}

interface RawThread {
  id: string;
  name: string;
  project_id?: string | null;
  embedding_model: string;
  settings?: ThreadSettings;
  thread_metadata?: ThreadMetadata;
  documents_meta?: Record<string, ThreadDocumentMeta>;
  created_at: string;
  message_count?: number;
  file_count?: number;
}

const mapThread = (raw: RawThread): Thread => ({
  id: raw.id,
  name: raw.name,
  project_id: raw.project_id,
  embeddingModel: raw.embedding_model,
  settings: raw.settings,
  thread_metadata: raw.thread_metadata,
  documents_meta: raw.documents_meta,
  created_at: raw.created_at,
  message_count: raw.message_count,
  file_count: raw.file_count,
});

export interface ThreadDocumentMeta {
  file_name?: string | null;
  page_count?: number | string | null;
  document_available_in_thread_at?: string | null;
  [key: string]: any;
}

export interface ThreadMetadata {
  fork?: {
    parent_thread_id?: string | null;
    parent_thread_name?: string | null;
    forked_at?: string | null;
    source_message_id?: string | null;
    source_message_created_at?: string | null;
    mode?: 'from_message' | 'full_thread' | string;
    memory_copy_mode?: string | null;
    copied_memory_ids?: string[];
  };
  fork_children?: string[];
  [key: string]: any;
}

export interface ThreadSettings {
  replans: number;
  system_role: string;
  tool_instructions: Record<string, string>;
  custom_instructions: string;
  hitl_web_approval: boolean;
  use_reranker: boolean;
  agent_workflow?: {
    workflow_id: string;
  };
  agent_workflow_validation?: {
    valid: boolean;
    code: string;
    requested_workflow_id: string;
    fallback_workflow_id: string;
  } | null;
  memory: {
    memory_enabled: boolean;
    thread_reads_thread_memory: boolean;
    thread_reads_project_memory: boolean;
    thread_reads_user_memory: boolean;
  };
}

export interface Project {
  id: string;
  name: string;
  description?: string | null;
  embeddingModel: string;
  settings_json?: Record<string, any>;
  created_at: string;
  last_activity_at: string;
  updated_at?: string | null;
}

export interface ProjectLifecycleSummary {
  project_id: string;
  thread_count: number;
  project_file_count: number;
  direct_file_count: number;
  unique_file_count: number;
  shared_file_count: number;
  orphan_file_count: number;
  memory_count: number;
  project_memory_count: number;
  thread_memory_count: number;
  memory_override_count: number;
  annotation_count: number;
  agent_run_count: number;
  active_run_count: number;
  protected: boolean;
  can_delete: boolean;
  can_clone: boolean;
  blocked_reason?: string | null;
}

export interface ProjectCloneResponse {
  project: Project;
  counts: Record<string, number>;
  warnings: Array<{ code: string; memory_id?: string; message: string }>;
}

export interface ProjectDeleteResponse {
  project_id: string;
  deleted: boolean;
  counts: Record<string, any>;
  warnings: string[];
}

interface RawProject {
  id: string;
  name: string;
  description?: string | null;
  embedding_model: string;
  settings_json?: Record<string, any>;
  created_at: string;
  last_activity_at: string;
  updated_at?: string | null;
}

const mapProject = (raw: RawProject): Project => ({
  id: raw.id,
  name: raw.name,
  description: raw.description,
  embeddingModel: raw.embedding_model,
  settings_json: raw.settings_json,
  created_at: raw.created_at,
  last_activity_at: raw.last_activity_at,
  updated_at: raw.updated_at,
});

export type MemoryScopeType = 'user' | 'project' | 'thread';

export interface MemoryAttributes {
  kind: 'preference' | 'profile' | 'instruction' | 'constraint' | 'decision' | 'fact';
  applicability: Array<'all_answers' | 'writing' | 'code' | 'research' | 'project' | 'task_specific'>;
  durability: 'stable' | 'time_sensitive';
}

export interface MemoryOverrideRef {
  id: string;
  scope_type: MemoryScopeType;
  scope_id: string;
  content: string;
  updated_at?: string | null;
}

export interface MemoryRecord {
  id: string;
  scope_type: MemoryScopeType;
  scope_id: string;
  content: string;
  attributes?: MemoryAttributes;
  embedding_model: string;
  content_hash?: string;
  index_status: 'pending' | 'indexing' | 'indexed' | 'failed' | string;
  index_attempts: number;
  indexed_at?: string | null;
  index_error?: string | null;
  source_refs?: Record<string, any>;
  source_refs_json?: Record<string, any>;
  overrides: MemoryOverrideRef[];
  overridden_by: MemoryOverrideRef[];
  created_at?: string | null;
  updated_at?: string | null;
  representations?: MemoryRepresentation[];
}

export interface MemoryRepresentation {
  embedding_model: string;
  primary: boolean;
  active?: boolean;
  index_status: string;
  index_attempts?: number;
  indexed_at?: string | null;
  index_error?: string | null;
}

export type MemoryResolutionStatus = 'effective' | 'overridden' | 'recall_disabled' | 'unavailable';

export interface MemoryWorkspaceRecord extends MemoryRecord {
  resolution_status: MemoryResolutionStatus;
  applied_overrides: MemoryOverrideRef[];
  applied_overridden_by: MemoryOverrideRef[];
}

export interface MemoryWorkspaceSection {
  scope_type: MemoryScopeType;
  scope_id: string;
  recall_enabled: boolean;
  recall_skip_reason?: string | null;
  memories: MemoryWorkspaceRecord[];
  truncated: boolean;
}

export type MemoryCuratorMode = 'create' | 'edit' | 'conversation_review' | 'memory_review';
export type MemoryCuratorState = 'clarification' | 'conflict' | 'proposal' | 'no_changes' | 'web_search_approval';

export interface MemoryCuratorWebSource {
  id: string;
  title: string;
  url: string;
  snippet: string;
  query: string;
  searched_at: string;
  score?: number;
}

export interface MemoryCuratorContext {
  selected_scope_type: MemoryScopeType;
  selected_scope_id: string;
  thread_id?: string | null;
  project_id?: string | null;
}

export interface MemoryCuratorMessage {
  role: 'user' | 'assistant';
  content: string;
  choice_id?: string;
}

export interface MemoryCuratorOperation {
  action: 'create' | 'update' | 'delete' | 'noop';
  scope_type?: MemoryScopeType;
  scope_id?: string;
  memory_id?: string;
  expected_updated_at?: string;
  content?: string;
  attributes?: MemoryAttributes;
  override_targets?: Array<{
    memory_id: string;
    expected_updated_at: string;
  }>;
  semantic_action?: 'create' | 'update' | 'delete' | 'move' | 'set_overrides';
  operation_group_id?: string;
  move_source_memory_id?: string;
  move_destination_memory_id?: string;
  web_sources?: Array<Omit<MemoryCuratorWebSource, 'snippet' | 'score'>>;
}

export interface MemoryOperationSummary {
  operation_group_id: string;
  action: 'create' | 'update' | 'delete' | 'move' | 'set_overrides';
  label: string;
  content?: string;
  attributes?: MemoryAttributes;
  source_memory_id?: string;
  source_scope?: { scope_type: MemoryScopeType; scope_id: string };
  destination_memory_id?: string;
  destination_scope?: { scope_type: MemoryScopeType; scope_id: string };
  override_target_ids: string[];
  removed_incoming_override_count: number;
}

export interface EffectiveMemoryResponse {
  context: { type: 'global' | 'project' | 'thread'; id: string; project_id?: string | null };
  policy: {
    requested_scopes: MemoryScopeType[];
    searched_scopes: Array<{ scope_type: MemoryScopeType; scope_id: string }>;
    skipped_scopes: Array<{ scope_type: MemoryScopeType; reason: string }>;
  };
  memories: MemoryRecord[];
  applied_overrides: Array<{ overriding_memory_id: string; overridden_memory_id: string }>;
  suppressed_memory_ids: string[];
  unavailable_memory_count: number;
  truncated: boolean;
  workspace_sections: MemoryWorkspaceSection[];
}

export interface MemoryReviewCursor {
  thread_id: string;
  reviewed_through_turn_id: string;
  reviewed_through_created_at: string;
}

export interface MemoryConsistencyReviewCursor {
  context_type: 'user' | 'project' | 'thread';
  context_id: string;
  snapshot_at: string;
  snapshot_scope_versions: Record<string, number>;
  anchor_position: number;
  reviewed_anchor_count: number;
  remaining_anchor_count: number;
}

export interface MemoryReviewStatus {
  context_type: 'user' | 'project' | 'thread';
  context_id: string;
  status: 'current' | 'review_suggested' | 'never_reviewed';
  embedding_model: string;
  current_scope_versions: Record<string, number>;
  reviewed_scope_versions: Record<string, number>;
  last_reviewed_at?: string | null;
}

export interface MemoryCuratorResponse {
  message: string;
  state: MemoryCuratorState;
  choices: Array<{
    id: string;
    label: string;
    description: string;
    user_message: string;
  }>;
  operations: MemoryCuratorOperation[];
  operation_summaries?: MemoryOperationSummary[];
  review?: {
    reviewed_count: number;
    remaining_count: number;
    cursor?: MemoryReviewCursor | null;
  } | null;
  memory_review?: (MemoryConsistencyReviewCursor & {
    candidate_groups: Array<{
      anchor_id: string;
      memories: Array<Pick<MemoryRecord, 'id' | 'scope_type' | 'scope_id' | 'content' | 'updated_at'> & { scope_rank: number }>;
    }>;
    representation_pending: boolean;
    missing_representation_count: number;
    blocked: boolean;
    embedding_model: string;
  }) | null;
  embedding_readiness: Array<{
    embedding_model: string;
    scopes?: Array<{ scope_type: MemoryScopeType; scope_id: string }>;
    ready: boolean;
    reason?: string;
  }>;
  consent?: {
    administration_available: boolean;
    memory_enabled?: boolean | null;
    thread_reads_thread_memory?: boolean | null;
    thread_reads_project_memory?: boolean | null;
    project_reads_user_memory?: boolean | null;
    thread_reads_user_memory?: boolean | null;
    effective_user_recall?: boolean | null;
  };
  context_memory_count?: number;
  tool_calls_used?: number;
  web_calls_used?: number;
  pending_web_search?: { query: string; reason: string } | null;
  web_sources?: MemoryCuratorWebSource[];
}

export interface MemoryChangeReceipt {
  operation_group_id: string;
  action: 'create' | 'update' | 'delete' | 'move' | 'set_overrides';
  source_memory_id?: string;
  result_memory_id?: string;
  source_scope?: { scope_type: MemoryScopeType; scope_id: string };
  destination_scope?: { scope_type: MemoryScopeType; scope_id: string };
  deleted_memory_ids: string[];
  override_target_ids: string[];
  removed_incoming_override_count: number;
  index_status?: string;
  warnings: Array<{ code: string; memory_id?: string; message: string }>;
}

export interface MemoryCuratorApplyResponse {
  changed_memories: MemoryRecord[];
  deleted_memory_ids: string[];
  warnings: Array<{ code: string; memory_id?: string; message: string }>;
  review_cursor_advanced: boolean;
  receipts: MemoryChangeReceipt[];
  memory_review_completed?: boolean;
  memory_review_cursor?: MemoryConsistencyReviewCursor | null;
}

export type MemoryManagerMode = 'direct_edit' | 'conversation_extract' | 'consistency_review';
export type MemoryManagerOperationType =
  | 'memory_create'
  | 'memory_update'
  | 'memory_delete'
  | 'memory_move'
  | 'memory_merge'
  | 'relationship_replace';

export interface MemoryManagerOperation {
  type: MemoryManagerOperationType;
  memory_id?: string;
  source_memory_id?: string;
  destination_memory_id?: string;
  scope_type?: MemoryScopeType;
  scope_id?: string;
  target_scope_type?: MemoryScopeType;
  target_scope_id?: string;
  content?: string;
  attributes?: MemoryAttributes;
  override_target_ids: string[];
  override_target_versions?: Record<string, string>;
  expected_updated_at?: string;
  operation_group_id?: string;
}

export interface MemoryManagerPlan {
  plan_id: string;
  plan_hash: string;
  mode: MemoryManagerMode;
  context: MemoryCuratorContext;
  state: 'proposal' | 'clarification' | 'no_changes' | 'blocked';
  message: string;
  choices: MemoryCuratorResponse['choices'];
  operations: MemoryManagerOperation[];
  analysis: MemoryOperationSummary[];
  review?: MemoryCuratorResponse['review'] | null;
  memory_review?: MemoryCuratorResponse['memory_review'] | null;
  budget: Record<string, number>;
  review_id?: string | null;
  next_cursor?: MemoryReviewCursor | MemoryConsistencyReviewCursor | null;
  scope_versions: Record<string, number>;
  embedding_readiness: MemoryCuratorResponse['embedding_readiness'];
  pending_web_search?: { query: string; reason: string } | null;
  web_sources: MemoryCuratorWebSource[];
  consent?: MemoryCuratorResponse['consent'];
}

export interface MemoryManagerApplyResponse extends MemoryCuratorApplyResponse {
  plan_id: string;
  plan_hash: string;
  idempotency_key: string;
  status: 'committed' | 'indexing_pending';
  review_id?: string | null;
}

// ============ Agent Workflow Builder API ============

export interface AgentWorkflowGraphSpec {
  nodes?: {
    id: string;
    type: string;
    [key: string]: any;
  }[];
  edges?: {
    from: string;
    to?: string;
    conditional?: boolean;
    routes?: Record<string, string>;
    [key: string]: any;
  }[];
}

export interface AgentWorkflowBuilderSpec {
  schema_version: 2;
  workflow_id: string;
  workflow_type: 'custom_rag_agent' | string;
  config: {
    graph?: AgentWorkflowGraphSpec;
    loop_policy?: Record<string, any>;
    hitl_policy?: Record<string, any>;
    allowed_tool_ids?: string[];
    tool_contract_ids?: string[];
    [key: string]: any;
  };
  [key: string]: any;
}

export interface AgentWorkflowContextPolicy {
  mode?: string;
  input_budget?: string;
  output_budget?: string;
  evidence_packet_limit?: number;
  evidence_packet_content_limit?: number;
  final_prompt_assembly?: string;
  [key: string]: any;
}

export interface AgentWorkflowObservability {
  span_kind?: string;
  event_prefix?: string;
  summary_fields?: string[];
  raw_payload?: string;
  [key: string]: any;
}

export interface AgentWorkflowNodeCatalogEntry {
  display_name: string;
  displayName?: string;
  category: string;
  capabilities: string[];
  allowed_route_functions: string[];
  allowed_tool_contract_ids: string[];
  allowed_parent_types: string[];
  allowed_child_types: string[];
  limits: Record<string, any>;
  state_reads: string[];
  state_writes: string[];
  prompt_slots: string[];
  context_policy: AgentWorkflowContextPolicy;
  observability: AgentWorkflowObservability;
  max_instances: number;
  ui?: {
    summary?: string;
    use_when?: string;
    category_label?: string;
    icon?: string;
    keywords?: string[];
    input_label?: string;
    output_label?: string;
    uses_llm?: boolean;
    uses_tools?: boolean;
    external_side_effect?: boolean;
    field_guidance?: Record<string, string>;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface AgentWorkflowRouteFunctionMetadata {
  id?: string;
  name?: string;
  display_name?: string;
  description?: string;
  allowed_source_node_types?: string[];
  route_labels?: string[];
  routes?: string[];
  route_options?: Record<string, {
    display_name?: string;
    description?: string;
    order?: number;
  }>;
  [key: string]: any;
}

export interface AgentWorkflowToolContract {
  id: string;
  category?: string;
  display_name?: string;
  description?: string;
  canonical_tools: string[];
  allowed_node_types: string[];
  required_node_capabilities: string[];
  artifact_keys: string[];
  warning_codes: string[];
  [key: string]: any;
}

export interface AgentWorkflowCatalogResponse {
  schema_version: number;
  spec_schema_version: 2 | number;
  graph_spec: {
    required_schema_version: 2 | number;
    requires_explicit_route_fn: boolean;
    reserved_node_ids: string[];
    start_node: string;
    end_node: string;
    [key: string]: any;
  };
  node_catalog: Record<string, AgentWorkflowNodeCatalogEntry>;
  route_functions: Record<string, AgentWorkflowRouteFunctionMetadata>;
  tool_contracts: Record<string, AgentWorkflowToolContract>;
  defaults: {
    context_policy?: AgentWorkflowContextPolicy;
    loop_policy?: Record<string, any>;
    parallel_policy?: {
      defaults: Record<string, boolean | number>;
      fields: Record<string, {
        type: 'boolean' | 'integer';
        default: boolean | number;
        minimum?: number;
        maximum?: number;
        step?: number;
        unit?: string;
        label: string;
      }>;
    };
    corrective_policy?: {
      defaults: Record<string, boolean | number | string>;
      fields: Record<string, {
        type: 'boolean' | 'integer' | 'number' | 'enum';
        default: boolean | number | string;
        minimum?: number;
        maximum?: number;
        step?: number;
        values?: string[];
        label: string;
      }>;
    };
    [key: string]: any;
  };
  [key: string]: any;
}

export interface AgentWorkflow {
  id: string;
  workflow_id?: string;
  builtin_key?: string | null;
  name: string;
  description?: string;
  visibility?: string;
  is_builtin?: boolean;
  is_default?: boolean;
  framework?: string;
  builder_id?: string;
  supports_replans?: boolean;
  supports_long_running_tasks?: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface AgentWorkflowValidationReport {
  valid: boolean;
  errors: string[];
  warnings: string[];
  issues?: AgentWorkflowValidationIssue[];
  schema_version?: number | null;
  [key: string]: any;
}

export interface AgentWorkflowValidationIssue {
  code: string;
  severity: 'error' | 'warning';
  message: string;
  node_id?: string | null;
  edge_index?: number | null;
  route?: string | null;
  allowed_alternatives?: string[];
  fix?: { kind: string; [key: string]: any } | null;
}

export interface AgentWorkflowSpecResponse {
  workflow_id: string;
  schema_version: number;
  spec_json: AgentWorkflowBuilderSpec | Record<string, any>;
  validation: AgentWorkflowValidationReport;
  validation_result_json?: Record<string, any>;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface InternalAgentWorkflowResponse {
  agent_workflow: AgentWorkflow;
  spec: AgentWorkflowSpecResponse;
}

export interface AgentWorkflowResponse {
  agent_workflow: AgentWorkflow;
  spec: AgentWorkflowSpecResponse;
  capabilities?: Record<string, any>;
}

export interface AgentWorkflowListResponse {
  agent_workflows: AgentWorkflow[];
}

export interface SaveInternalAgentWorkflowPayload {
  workflow_id?: string;
  name: string;
  description?: string;
  spec_json: AgentWorkflowBuilderSpec | Record<string, any>;
  framework?: string;
  builder_id?: string;
}

export interface ThreadAgentConfigValidationResponse {
  valid: boolean;
  workflow_id: string;
  validation: AgentWorkflowValidationReport;
  resolved_spec_json: AgentWorkflowBuilderSpec | Record<string, any>;
}

const readApiError = async (res: Response): Promise<string> => {
  const text = await res.text();
  if (!text) return `${res.status} ${res.statusText}`.trim();
  try {
    const parsed = JSON.parse(text);
    const detail = parsed?.detail;
    if (typeof detail === 'string') return detail;
    if (detail?.message) return String(detail.message);
    return JSON.stringify(parsed);
  } catch {
    return text;
  }
};

export interface PromptToolDefinition {
  id: string;
  display_name: string;
  description: string;
  default_prompt: string;
}

export interface PromptDefaults {
  replans_limit: number;
  context_window: number;
  system_role: string;
  tool_instructions: Record<string, string>;
  custom_instructions: string;
  hitl_web_approval?: boolean;
  use_reranker?: boolean;
  agent_workflow?: {
    workflow_id: string;
  };
}

export interface ThreadFile {
  fileHash: string;
  fileName: string;
  filePath?: string;
  sourceType?: ThreadFileSourceTypeValue;
  addedAt?: string;
  associationScope?: "thread" | "project";
  isProjectKnowledge?: boolean;
  processingStatus?: "pending" | "completed" | "failed";
  processingError?: string;
}

interface RawThreadFile {
  file_hash: string;
  file_name: string;
  file_path?: string;
  source_type?: ThreadFileSourceTypeValue;
  added_at?: string;
  association_scope?: "thread" | "project";
  is_project_knowledge?: boolean;
  processing_status?: "pending" | "completed" | "failed";
  processing_error?: string;
}

const mapThreadFile = (raw: RawThreadFile): ThreadFile => ({
  fileHash: raw.file_hash,
  fileName: raw.file_name,
  filePath: raw.file_path,
  sourceType: raw.source_type,
  addedAt: raw.added_at,
  associationScope: raw.association_scope,
  isProjectKnowledge: raw.is_project_knowledge,
  processingStatus: raw.processing_status,
  processingError: raw.processing_error,
});

export interface WebSource {
  text: string;
  url: string;
  title: string;
  score?: number;
}

export interface AgentMessageMetadata {
  agent_run_id?: string;
  agent_workflow_id?: string;
  agent_route?: string;
  agent_route_reason?: string;
}

export interface AgentTraceRefs {
  node_ids?: string[];
  span_ids?: string[];
  interrupt_id?: string | null;
  [key: string]: any;
}

export interface Message {
  id: string;
  role: MessageRoleValue;
  content: string;
  created_at: string;
  isRecollected?: boolean;
  reasoning?: string;
  reasoning_available?: boolean;
  reasoning_format?: ReasoningFormatValue;
  context_compact?: string;
  web_sources?: WebSource[];
  metadata?: AgentMessageMetadata;
  agent_run_id?: string;
  agent_run_turn_kind?: string;
  agent_run_sequence?: number | null;
  agent_trace_refs?: AgentTraceRefs | null;
  agent_workflow_id?: string;
  agent_route?: string;
  agent_route_reason?: string;
}

export interface AgentTraceEvent {
  name: string;
  attributes?: Record<string, any>;
  input?: Record<string, any>;
  output?: Record<string, any>;
  [key: string]: any;
}

export interface AgentTraceSpan {
  span_id: string;
  parent_span_id?: string | null;
  name: string;
  kind: 'AGENT' | 'CHAIN' | 'LLM' | 'RETRIEVER' | 'TOOL' | 'PROMPT' | string;
  status?: string;
  start_time?: string | null;
  end_time?: string | null;
  duration_ms?: number | null;
  attributes?: Record<string, any>;
  input?: Record<string, any>;
  output?: Record<string, any>;
  events?: AgentTraceEvent[];
  links?: Record<string, any>[];
  raw?: Record<string, any>;
  [key: string]: any;
}

export interface AgentDebugTrace {
  schema_version: number;
  trace_id: string;
  run_id: string;
  thread_id?: string;
  chat_turn_id?: string | null;
  user_id?: string | null;
  workflow_id?: string;
  workflow_type?: string;
  status?: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
  attributes?: Record<string, any>;
  metrics?: Record<string, any>;
  spans?: AgentTraceSpan[];
  links?: Record<string, any>[];
  artifacts?: Record<string, any>[];
  [key: string]: any;
}

export interface AgentDebugSummary {
  status?: string;
  route?: string;
  routeReason?: string;
  durationMs?: number | null;
  metrics?: Record<string, any>;
  nodes?: Record<string, any>[];
  operations?: Record<string, any>[];
  tools?: Record<string, any>[];
  usedNodeCount?: number;
  availableNodeCount?: number | null;
  usedToolCount?: number;
  availableToolCount?: number | null;
  warningCount?: number;
  errorCount?: number;
  errors?: Record<string, any>[];
  [key: string]: any;
}

export interface AgentRunDebug {
  version?: number;
  trace?: AgentDebugTrace;
  summary?: AgentDebugSummary;
  graph?: {
    nodes?: Record<string, any>[];
    edges?: Record<string, any>[];
    executionPlan?: string[];
    selectedRoute?: string;
    [key: string]: any;
  };
  detail_manifest?: AgentRunNodeDetailManifest[];
  detail_safety?: Record<string, any>;
  final_output?: AgentRunFinalOutput;
  topology?: {
    available?: boolean;
    kind?: string | null;
    operation_refs?: boolean;
  };
}

export interface AgentRunNodeDetailManifest {
  node_id: string;
  node_type?: string;
  visit_index: number;
  status?: string;
  available: boolean;
  size_bytes?: number;
  truncated?: boolean;
}

export interface AgentRunFinalOutput {
  answer?: string;
  clarification_options?: string[];
  route?: string;
  route_reason?: string;
  reasoning?: string;
  reasoning_available?: boolean;
  reasoning_format?: string;
  safety?: Record<string, any>;
}

export interface AgentRunNodeDetail {
  node_id: string;
  node_type?: string;
  visit_index: number;
  status?: string;
  checkpoint_before?: Record<string, any>;
  changes?: Record<string, any>;
  checkpoint_after?: Record<string, any>;
  output?: Record<string, any>;
  event?: Record<string, any>;
  llm?: {
    prompt?: { role?: string; content?: unknown }[];
    response?: unknown;
    reasoning?: string;
    reasoning_available?: boolean;
    reasoning_format?: string;
  };
  tools?: Record<string, any>[];
  error?: unknown;
  safety?: Record<string, any>;
  [key: string]: any;
}

export type AgentRunResumeAction = AgentRunResumeActionValue;

export interface AgentRunPendingInterrupt {
  interrupt_id: string;
  gate_id?: string | null;
  node_id?: string | null;
  type?: string | null;
  status?: InterruptStatusValue | string;
  requested_at?: string | null;
  expires_at?: string | null;
  default_action?: AgentRunResumeActionValue | string | null;
  allowed_actions?: AgentRunResumeActionValue[] | string[];
  mode?: HitlModeValue | string | null;
  phase?: HitlPhaseValue | string | null;
  selection_mode?: HitlSelectionModeValue | string | null;
  options?: {
    id: string;
    label?: string;
    target_node_id?: string;
    description?: string;
    [key: string]: any;
  }[];
  prompt?: string | null;
  title?: string | null;
  body?: string | null;
  input_summary?: Record<string, any> | string | null;
  proposed_action?: Record<string, any> | string | null;
  proposed_tool?: Record<string, any> | string | null;
  proposed_memory?: Record<string, any> | string | null;
  proposed_final_answer?: string | null;
  resume_token?: string | null;
  resume_version?: number | null;
  decision?: Record<string, any> | null;
  [key: string]: any;
}

export interface AgentRunDetails {
  id: string;
  thread_id: string;
  status: string;
  workflow_id: string;
  task_id?: string | null;
  parent_run_id?: string | null;
  task_attempt?: number;
  resolved_spec_json?: Record<string, any>;
  metrics_json?: Record<string, any>;
  parallel_summary?: {
    dispatch_id?: string;
    planned?: number;
    completed?: number;
    skipped?: number;
    failed?: number;
    timed_out?: number;
    cancelled?: number;
    retried?: number;
    partial_evidence?: boolean;
    elapsed_ms?: number;
  } | null;
  error_json?: Record<string, any> | null;
  started_at?: string;
  completed_at?: string | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  debug?: AgentRunDebug | null;
  final_output?: AgentRunFinalOutput | null;
  turns?: {
    id: string;
    kind?: string | null;
    sequence?: number | null;
    trace_refs?: AgentTraceRefs | null;
  }[];
  [key: string]: any;
}

export async function getAgentRunNodeDetails(
  runId: string,
  threadId: string,
  nodeId: string,
  visitIndex: number,
): Promise<AgentRunNodeDetail> {
  const params = new URLSearchParams({ thread_id: threadId, node_id: nodeId, visit_index: String(visitIndex) });
  const res = await fetch(`${API_BASE}/api/agent-runs/${encodeURIComponent(runId)}/details?${params.toString()}`);
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()).detail;
}

export async function getAgentRunOperationDetails(
  runId: string,
  threadId: string,
  operationId: string,
  visitIndex: number,
): Promise<AgentRunNodeDetail> {
  const params = new URLSearchParams({ thread_id: threadId, visit_index: String(visitIndex) });
  const res = await fetch(`${API_BASE}/api/agent-runs/${encodeURIComponent(runId)}/operations/${encodeURIComponent(operationId)}/details?${params.toString()}`);
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()).detail;
}

export interface AgentRunSummary {
  id: string;
  thread_id: string;
  workflow_id: string;
  task_id?: string | null;
  parent_run_id?: string | null;
  task_attempt?: number;
  status: string;
  started_at?: string | null;
  completed_at?: string | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  metrics?: Record<string, any>;
  error?: Record<string, any> | null;
  [key: string]: any;
}

export type RuntimeSupportLevel = 'native' | 'emulated' | 'conditional' | 'unsupported';

export interface RuntimeOperationDescriptor {
  support: RuntimeSupportLevel;
  enabled: boolean;
  disabled_reason?: string | null;
  modes?: string[];
  semantics?: string | null;
  confirmation?: string | null;
  terminal_states?: string[];
  preserves_run_id?: boolean | null;
  preserves_session_id?: boolean | null;
}

export interface RuntimeCapabilities {
  operations: Record<string, RuntimeOperationDescriptor>;
  runtime_version?: string | null;
  contract_version: number;
}

export interface AgentRuntimeCapabilityResponse {
  resource: 'deployment' | 'definition' | 'run';
  runtime_id: string;
  framework: string;
  builder_id: string;
  available: boolean;
  capabilities: RuntimeCapabilities | null;
  definition_id?: string;
  run_id?: string;
  run_status?: string;
  error?: Record<string, any>;
}

export interface AgentRuntimeListResponse {
  agent_runtimes: AgentRuntimeCapabilityResponse[];
}

export async function listProjects(): Promise<{ projects: Project[] }> {
  const res = await fetch(`${API_BASE}/api/projects`);
  if (!res.ok) throw new Error(await res.text());
  const raw = await res.json();
  return { projects: (raw.projects || []).map(mapProject) };
}

export async function getProject(projectId: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}`);
  if (!res.ok) throw new Error(await res.text());
  return mapProject(await res.json());
}

export interface MemoryWorkspaceReadiness {
  context_type: 'global' | 'project' | 'thread';
  thread_id?: string | null;
  project_id?: string | null;
  embedding_model: string;
  embedding_model_ready: boolean;
  status: 'ready' | 'indexing' | 'blocked' | 'error';
  ready: boolean;
  canonical: Record<string, number>;
  global_representations: {
    embedding_model: string;
    ready: boolean;
    total_count: number;
    indexed_count: number;
    pending_count: number;
    failed_count: number;
  };
}

const memoryWorkspaceReadinessPath = (
  action: 'prepare' | 'status',
  input: { threadId?: string | null; projectId?: string | null },
) => {
  const params = new URLSearchParams();
  if (input.threadId) params.set('thread_id', input.threadId);
  if (input.projectId) params.set('project_id', input.projectId);
  const query = params.toString();
  return `/api/memory-workspace/${action}${query ? `?${query}` : ''}`;
};

export async function prepareMemoryWorkspace(input: {
  threadId?: string | null;
  projectId?: string | null;
}): Promise<MemoryWorkspaceReadiness> {
  const res = await fetch(`${API_BASE}${memoryWorkspaceReadinessPath('prepare', input)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getMemoryWorkspaceStatus(input: {
  threadId?: string | null;
  projectId?: string | null;
}): Promise<MemoryWorkspaceReadiness> {
  const res = await fetch(`${API_BASE}${memoryWorkspaceReadinessPath('status', input)}`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function createProject(
  name: string,
  embeddingModel: string,
  description?: string,
  settingsJson?: Record<string, any>
): Promise<Project> {
  const res = await fetch(`${API_BASE}/api/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      embedding_model: embeddingModel,
      description,
      settings_json: settingsJson,
    })
  });
  if (!res.ok) throw new Error(await res.text());
  return mapProject(await res.json());
}

export async function updateProject(
  projectId: string,
  updates: { name?: string; description?: string; settings_json?: Record<string, any> }
): Promise<Project> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error(await res.text());
  return mapProject(await res.json());
}

export async function getProjectLifecycleSummary(
  projectId: string
): Promise<ProjectLifecycleSummary> {
  const res = await fetch(
    `${API_BASE}/api/projects/${encodeURIComponent(projectId)}/lifecycle-summary`
  );
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function cloneProject(
  projectId: string,
  name: string,
  includeThreads: boolean
): Promise<ProjectCloneResponse> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}/clone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, include_threads: includeThreads }),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  const raw = await res.json();
  return {
    ...raw,
    project: mapProject(raw.project),
  };
}

export async function deleteProject(projectId: string): Promise<ProjectDeleteResponse> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function createThread(name: string, projectId?: string | null): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, project_id: projectId || undefined })
  });
  if (!res.ok) throw new Error(await res.text());
  return mapThread(await res.json());
}

export async function createProjectThread(projectId: string, name: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/projects/${encodeURIComponent(projectId)}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
  if (!res.ok) throw new Error(await res.text());
  return mapThread(await res.json());
}

export async function listMemories(
  scopeType: MemoryScopeType,
  scopeId: string,
  limit = 500
): Promise<{ memories: MemoryRecord[] }> {
  const params = new URLSearchParams({
    scope_type: scopeType,
    scope_id: scopeId,
    limit: String(limit),
  });
  const res = await fetch(`${API_BASE}/api/memories?${params.toString()}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listEffectiveMemories(input: {
  threadId?: string | null;
  projectId?: string | null;
  limit?: number;
}): Promise<EffectiveMemoryResponse> {
  const limit = input.limit ?? 500;
  const path = input.threadId
    ? `/api/threads/${encodeURIComponent(input.threadId)}/memories/effective`
    : input.projectId
      ? `/api/projects/${encodeURIComponent(input.projectId)}/memories/effective`
      : '/api/memories/effective';
  const res = await fetch(`${API_BASE}${path}?limit=${encodeURIComponent(String(limit))}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteMemory(memoryId: string): Promise<{
  status: string;
  memory_id: string;
  vector_cleanup: string;
}> {
  const res = await fetch(`${API_BASE}/api/memories/${encodeURIComponent(memoryId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function retryMemoryIndex(memoryId: string, embeddingModel?: string): Promise<MemoryRecord> {
  const query = embeddingModel ? `?embedding_model=${encodeURIComponent(embeddingModel)}` : '';
  const res = await fetch(`${API_BASE}/api/memories/${encodeURIComponent(memoryId)}/index${query}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function planMemoryManager(input: {
  mode: MemoryManagerMode;
  context: MemoryCuratorContext;
  memory_id?: string;
  messages: MemoryCuratorMessage[];
  llm_model: string;
  context_window: number;
  review_round?: number;
  review_id?: string | null;
  memory_review_cursor?: MemoryConsistencyReviewCursor | null;
  web_search_mode: 'off' | 'ask' | 'on';
  web_search_decision?: { query: string; approved: boolean };
}): Promise<MemoryManagerPlan> {
  const res = await fetch(`${API_BASE}/api/memory-manager/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function applyMemoryManagerPlan(input: {
  plan: MemoryManagerPlan;
  idempotency_key: string;
  actor_id?: string;
}): Promise<MemoryManagerApplyResponse> {
  const res = await fetch(`${API_BASE}/api/memory-manager/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      plan: input.plan,
      plan_hash: input.plan.plan_hash,
      idempotency_key: input.idempotency_key,
      confirmed: true,
      actor_id: input.actor_id || 'ui',
    }),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getMemoryReviewStatus(input: {
  threadId?: string | null;
  projectId?: string | null;
}): Promise<MemoryReviewStatus> {
  const path = input.threadId
    ? `/api/threads/${encodeURIComponent(input.threadId)}/memories/review-status`
    : input.projectId
      ? `/api/projects/${encodeURIComponent(input.projectId)}/memories/review-status`
      : '/api/memories/review-status';
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function listThreads(): Promise<{ threads: Thread[] }> {
  const res = await fetch(`${API_BASE}/api/threads`);
  if (!res.ok) throw new Error(await res.text());
  const raw = await res.json();
  return { threads: (raw.threads || []).map(mapThread) };
}

export async function getThread(threadId: string): Promise<Thread & {
  files: ThreadFile[];
  stats: any;
  embeddingModelReady?: boolean;
  stats_unavailable_reason?: string | null;
}> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}`);
  if (!res.ok) throw new Error(await res.text());
  const raw = await res.json();
  return {
    ...mapThread(raw),
    files: (raw.files || []).map(mapThreadFile),
    stats: raw.stats,
    embeddingModelReady: raw.embedding_model_ready,
    stats_unavailable_reason: raw.stats_unavailable_reason,
  };
}

export async function updateThread(threadId: string, name: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
  if (!res.ok) throw new Error(await res.text());
  return mapThread(await res.json());
}

export async function forkThread(
  threadId: string,
  options: { messageId?: string; name?: string; targetProjectId?: string; memoryCopyMode?: string } = {}
): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message_id: options.messageId,
      name: options.name,
      target_project_id: options.targetProjectId,
      memory_copy_mode: options.memoryCopyMode,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return mapThread(await res.json());
}

export async function getThreadSettings(threadId: string): Promise<ThreadSettings> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/settings`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateThreadSettings(
  threadId: string,
  settings: Partial<ThreadSettings>
): Promise<ThreadSettings> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getPromptTools(): Promise<{ tools: PromptToolDefinition[]; defaults: PromptDefaults }> {
  const res = await fetch(`${API_BASE}/api/threads/prompt-tools`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getPromptPreview(payload: {
  context_window: number;
  system_role: string;
  tool_instructions: Record<string, string>;
  custom_instructions: string;
  use_web_search?: boolean;
  agent_workflow_id?: string;
}): Promise<{ prompt: string }> {
  const res = await fetch(`${API_BASE}/api/threads/prompt-preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      ...getBrowserRuntimeContext(),
    })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getInternalAgentWorkflowCatalog(): Promise<AgentWorkflowCatalogResponse> {
  const res = await fetch(`${API_BASE}/api/internal/agent-workflows/catalog`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function listAgentWorkflows(): Promise<AgentWorkflowListResponse> {
  const res = await fetch(`${API_BASE}/api/agent-workflows`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function listAgentRuntimes(): Promise<AgentRuntimeListResponse> {
  const res = await fetch(`${API_BASE}/api/agent-runtimes`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getAgentRuntimeCapabilities(
  runtimeId: string,
): Promise<AgentRuntimeCapabilityResponse> {
  const res = await fetch(`${API_BASE}/api/agent-runtimes/${encodeURIComponent(runtimeId)}/capabilities`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getAgentWorkflow(
  workflowId: string,
): Promise<AgentWorkflowResponse> {
  const res = await fetch(`${API_BASE}/api/agent-workflows/${encodeURIComponent(workflowId)}`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getAgentWorkflowCapabilities(
  workflowId: string,
): Promise<AgentRuntimeCapabilityResponse> {
  const res = await fetch(`${API_BASE}/api/agent-workflows/${encodeURIComponent(workflowId)}/capabilities`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getBuiltinAgentWorkflowSource(
  builtinKey: string,
): Promise<{ builtin_key: string; name: string; description: string; spec_json: AgentWorkflowBuilderSpec | Record<string, any> }> {
  const res = await fetch(`${API_BASE}/api/agent-workflows/builtins/${encodeURIComponent(builtinKey)}/source`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function saveInternalAgentWorkflow(
  payload: SaveInternalAgentWorkflowPayload
): Promise<InternalAgentWorkflowResponse> {
  const res = await fetch(`${API_BASE}/api/internal/agent-workflows`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function getInternalAgentWorkflow(
  workflowId: string
): Promise<InternalAgentWorkflowResponse> {
  const res = await fetch(`${API_BASE}/api/internal/agent-workflows/${encodeURIComponent(workflowId)}`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function deleteInternalAgentWorkflow(
  workflowId: string
): Promise<{ status: string; agent_workflow: AgentWorkflow }> {
  const res = await fetch(`${API_BASE}/api/internal/agent-workflows/${encodeURIComponent(workflowId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function validateAgentWorkflowSpec(
  spec: AgentWorkflowBuilderSpec | Record<string, any>
): Promise<AgentWorkflowValidationReport> {
  const res = await fetch(`${API_BASE}/api/agent-workflows/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ spec }),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function validateThreadAgentConfig(
  threadId: string,
  overrides: Record<string, any> = {}
): Promise<ThreadAgentConfigValidationResponse> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/agent-config/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ overrides }),
  });
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}`, {
    method: "DELETE"
  });
  if (!res.ok) throw new Error(await res.text());
}

export interface BulkDeleteThreadsResponse {
  deleted_thread_ids: string[];
  not_found_thread_ids: string[];
  failed_thread_ids: { thread_id: string; error: string }[];
}

export async function bulkDeleteThreads(threadIds: string[]): Promise<BulkDeleteThreadsResponse> {
  const res = await fetch(`${API_BASE}/api/threads/bulk/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_ids: threadIds }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function addFileToThread(
  threadId: string,
  fileHash: string,
  fileName: string,
  text?: string
): Promise<any> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_hash: fileHash,
      file_name: fileName,
      text
    })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThreadFiles(threadId: string): Promise<{ files: ThreadFile[] }> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files`);
  if (!res.ok) throw new Error(await res.text());
  const raw = await res.json();
  return { files: (raw.files || []).map(mapThreadFile) };
}

export async function getProjectFiles(projectId: string): Promise<{ files: ThreadFile[] }> {
  const res = await fetch(`${API_BASE}/api/projects/${projectId}/files`);
  if (!res.ok) throw new Error(await readApiError(res));
  const raw = await res.json();
  return { files: (raw.files || []).map(mapThreadFile) };
}

export async function promoteFileToProject(
  projectId: string,
  file: Pick<ThreadFile, "fileHash" | "fileName" | "filePath">,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/projects/${projectId}/files`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_hash: file.fileHash, file_name: file.fileName, file_path: file.filePath }),
  });
  if (!res.ok) throw new Error(await readApiError(res));
}

export async function removeSourceFromProject(projectId: string, fileHash: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/projects/${projectId}/files/${fileHash}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await readApiError(res));
}

export async function retryProjectFile(projectId: string, fileHash: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/projects/${projectId}/files/${fileHash}/retry`, { method: "POST" });
  if (!res.ok) throw new Error(await readApiError(res));
}

export async function retryTargetFile(target: KnowledgeTarget, fileHash: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/${targetPath(target)}/files/${fileHash}/retry`, { method: "POST" });
  if (!res.ok) throw new Error(await readApiError(res));
}

export interface ThreadFileAnnotationsResponse {
  thread_id: string;
  file_hash: string;
  annotations: AnnotationTransferItem[];
  created_at?: string;
  updated_at?: string;
}

export async function getThreadFileAnnotations(
  threadId: string,
  fileHash: string
): Promise<ThreadFileAnnotationsResponse> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/${fileHash}/annotations`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return {
    ...data,
    annotations: deserializeAnnotationItems((data.annotations || []) as AnnotationTransferItem[]),
  };
}

export async function updateThreadFileAnnotations(
  threadId: string,
  fileHash: string,
  annotations: AnnotationTransferItem[]
): Promise<ThreadFileAnnotationsResponse> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/${fileHash}/annotations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      annotations: serializeAnnotationItems(annotations),
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return {
    ...data,
    annotations: deserializeAnnotationItems((data.annotations || []) as AnnotationTransferItem[]),
  };
}

export async function captureBrowserPage(threadId: string): Promise<{
  status: string;
  fileHash: string;
  url: string;
  title: string;
  indexing: string;
  ready?: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/browser-capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error);
  }
  const raw = await res.json();
  return {
    status: raw.status,
    fileHash: raw.file_hash,
    url: raw.url,
    title: raw.title,
    indexing: raw.indexing,
    ready: raw.ready,
  };
}

export async function captureBrowserPageForTarget(target: KnowledgeTarget): Promise<{
  status: string; fileHash: string; url: string; title: string; indexing: string; ready?: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/${targetPath(target)}/browser-capture`, { method: "POST" });
  if (!res.ok) throw new Error(await readApiError(res));
  const raw = await res.json();
  return {
    status: raw.status,
    fileHash: raw.file_hash,
    url: raw.url,
    title: raw.title,
    indexing: raw.indexing,
    ready: raw.ready,
  };
}

export async function getPdfForTarget(fileHash: string, target: KnowledgeTarget): Promise<PdfData> {
  const res = await fetch(`${API_BASE}/api/${targetPath(target)}/files/${fileHash}`);
  if (!res.ok) throw new Error(await readApiError(res));
  return mapPdfData(await res.json());
}

export async function getParsedSentencesForTarget(fileHash: string, target: KnowledgeTarget) {
  const res = await fetch(`${API_BASE}/api/${targetPath(target)}/files/${fileHash}/sentences`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function pollForTargetFileReady(
  target: KnowledgeTarget,
  fileHash: string,
  options: { maxAttempts?: number; intervalMs?: number; timeoutMs?: number } = {},
): Promise<boolean> {
  const { maxAttempts = 10, intervalMs = 500, timeoutMs = 5000 } = options;
  const started = Date.now();
  for (let attempt = 0; attempt < maxAttempts && Date.now() - started <= timeoutMs; attempt += 1) {
    try {
      const res = await fetch(`${API_BASE}/api/${targetPath(target)}/files/${fileHash}/download`, { method: "HEAD" });
      if (res.ok) return true;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return false;
}

/**
 * Poll for file ready status using HEAD request.
 * Returns true when file is confirmed accessible, false on timeout.
 */
export async function pollForFileReady(
  threadId: string,
  fileHash: string,
  options: {
    maxAttempts?: number;
    intervalMs?: number;
    timeoutMs?: number;
  } = {}
): Promise<boolean> {
  const { maxAttempts = 10, intervalMs = 500, timeoutMs = 5000 } = options;
  const startTime = Date.now();

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Check timeout
    if (Date.now() - startTime > timeoutMs) {
      return false;
    }

    try {
      const res = await fetch(
        `${API_BASE}/api/threads/${threadId}/files/${fileHash}/download`,
        { method: "HEAD" }
      );
      if (res.ok) {
        return true;
      }
    } catch (e) {
      // Network error, will retry
    }

    // Wait before next attempt
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  return false;
}

export async function removeSourceFromThread(
  threadId: string,
  fileHash: string
): Promise<{ status: string; removed_from_db: boolean }> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/${fileHash}`, {
    method: "DELETE"
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThreadMessages(
  threadId: string,
  limit: number = 100,
  offset: number = 0
): Promise<{ messages: Message[] }> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/messages?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteMessage(messageId: string): Promise<{ deleted_ids: string[] }> {
  const res = await fetch(`${API_BASE}/api/messages/${messageId}`, {
    method: "DELETE"
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listThreadAgentRuns(
  threadId: string,
  options: { limit?: number; status?: string } = {}
): Promise<{ thread_id: string; limit: number; status?: string | null; agent_runs: AgentRunSummary[] }> {
  const params = new URLSearchParams();
  params.set("limit", String(options.limit ?? 20));
  if (options.status) params.set("status", options.status);
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/agent-runs?${params.toString()}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getAgentRun(runId: string, threadId: string): Promise<AgentRunDetails> {
  const params = new URLSearchParams({ thread_id: threadId });
  const res = await fetch(`${API_BASE}/api/agent-runs/${runId}?${params.toString()}`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.agent_run;
}

export async function getAgentRunCapabilities(
  runId: string,
  threadId: string,
): Promise<AgentRuntimeCapabilityResponse> {
  const params = new URLSearchParams({ thread_id: threadId });
  const res = await fetch(`${API_BASE}/api/agent-runs/${encodeURIComponent(runId)}/capabilities?${params.toString()}`);
  if (!res.ok) throw new Error(await readApiError(res));
  return res.json();
}

export async function cancelChatAgentRun(
  runId: string,
  threadId: string,
): Promise<{ status: 'cancel_requested' | 'already_terminal'; run_id?: string; run_status?: string }> {
  const res = await fetch(`${API_BASE}/api/agent-runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface AgentRunResumePayload {
  action: AgentRunResumeAction;
  interrupt_id: string;
  edited_payload?: Record<string, any>;
  client_metadata?: Record<string, any>;
  selected_option_ids?: string[];
  resume_token?: string;
  resume_version?: number;
  thread_id?: string;
  runtime_approval_choice?: 'once' | 'session' | 'always' | 'deny';
  resolve_all?: boolean;
}

export interface AgentRunResumeResponse {
  agent_run: AgentRunDetails;
  interrupt: AgentRunPendingInterrupt;
  outcome: string;
  duplicate: boolean;
}

export async function resumeAgentRun(
  runId: string,
  payload: AgentRunResumePayload,
): Promise<AgentRunResumeResponse> {
  const res = await fetch(`${API_BASE}/api/agent-runs/${runId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function streamResumeAgentRun(
  runId: string,
  payload: AgentRunResumePayload,
  onEvent: (event: AgentExecutionStreamEnvelope) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/agent-runs/${encodeURIComponent(runId)}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  });
  await consumeAgentExecutionStream(res, onEvent);
}

// ============ Durable Deep Research Tasks ============

export type AgentTaskStatus = 'created' | 'queued' | 'running' | 'pausing' | 'paused' | 'awaiting_approval' | 'cancelling' | 'cancelled' | 'completed' | 'failed' | 'expired';

export interface AgentTaskSummary {
  id: string;
  thread_id: string;
  objective: string;
  workflow_id: string;
  status: AgentTaskStatus;
  version: number;
  active_run_id?: string | null;
  run_attempt?: number;
  progress: number;
  web_access?: 'undecided' | 'allowed_for_task' | 'denied_for_task';
  completed_todos: number;
  total_todos: number;
  current_phase: string;
  terminal_reason?: string | null;
  budgets: Record<string, number>;
  configuration: Record<string, any>;
  created_at: string;
  updated_at: string;
  active_run?: { id: string; status: string; checkpoint_thread_id?: string; pending_interrupt?: AgentRunPendingInterrupt | null } | null;
  plan?: { revision: number; reason: string; objective: string; completion_criteria: string[]; ordered_todo_ids: string[]; content_hash: string } | null;
}

export interface AgentTaskTodo {
  id: string;
  title: string;
  description: string;
  completion_criteria: string;
  status: string;
  priority: number;
  required: boolean;
  dependency_ids: string[];
  profile_id: string;
  attempt: number;
  max_attempts: number;
  progress: number;
  result_summary?: string | null;
  terminal_reason?: string | null;
  artifact_ids: string[];
}

export interface AgentTaskArtifact {
  id: string;
  run_id: string;
  kind: string;
  media_type: string;
  byte_size: number;
  sha256: string;
  todo_id?: string | null;
  validity: string;
  sensitivity: string;
  created_at: string;
}

export interface AgentTaskRun {
  id: string;
  task_id: string;
  attempt: number;
  parent_run_id?: string | null;
  status: string;
  checkpoint_thread_id?: string | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  metrics: Record<string, any>;
  error?: Record<string, any> | null;
  debug?: AgentRunDebug | null;
  started_at: string;
  completed_at?: string | null;
}

export type AgentTaskTimelineType = 'objective' | 'plan' | 'todo_result' | 'todo_failure' | 'run_failure' | 'approval' | 'replan' | 'final_report';

export interface AgentTaskTimelineSource {
  id: string;
  kind: 'web' | 'document' | 'memory' | 'thread';
  title?: string;
  url?: string;
  snippet?: string;
  file_hash?: string;
  page?: number;
  memory_id?: string;
  artifact_id?: string;
  origin_run_id: string;
  origin_attempt: number;
  plan_revision: number;
  inherited?: boolean;
  origins: Array<{
    run_id: string;
    attempt: number;
    artifact_id: string;
    plan_revision: number;
    inherited: boolean;
  }>;
}

export interface AgentTaskTimelineItem {
  id: string;
  type: AgentTaskTimelineType;
  status: string;
  primary_content: string;
  timestamp?: string | null;
  folds?: Record<string, any>;
  artifact_ids?: string[];
  artifacts?: AgentTaskArtifact[];
  sources?: AgentTaskTimelineSource[];
  evidence_manifest?: Array<Record<string, any>>;
  trace_anchor?: Record<string, any> | null;
}

export interface AgentTaskSubagentRun {
  id: string;
  todo_id: string;
  profile_id: string;
  attempt: number;
  status: string;
  timeout_ms: number;
  usage: Record<string, number>;
  error?: Record<string, any> | null;
  output_artifact_ids: string[];
}

const taskQuery = (threadId: string) => new URLSearchParams({ thread_id: threadId }).toString();

export type DeepResearchEngine = 'langgraph' | 'hermes';

export async function getDeepResearchCapabilities(): Promise<{
  enabled: boolean;
  web_enabled: boolean;
  limits: Record<string, number>;
  engines: Record<DeepResearchEngine, { enabled: boolean; workflow_id: string; max_context_length?: number | null }>;
}> {
  const response = await fetch(`${API_BASE}/api/deep-research/capabilities`);
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}

export async function createAgentTask(threadId: string, payload: {
  objective: string; llm_model: string; context_window: number; web_search_mode: 'off' | 'ask' | 'on'; engine?: DeepResearchEngine;
}): Promise<AgentTaskSummary> {
  const response = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}/agent-tasks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).task;
}

export async function listAgentTasks(threadId: string): Promise<AgentTaskSummary[]> {
  const response = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}/agent-tasks`);
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).tasks;
}

export async function getAgentTask(taskId: string, threadId: string): Promise<AgentTaskSummary> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}?${taskQuery(threadId)}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).task;
}

export async function commandAgentTask(taskId: string, threadId: string, action: 'start' | 'pause' | 'resume' | 'cancel' | 'retry', version: number): Promise<AgentTaskSummary> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/${action}?${taskQuery(threadId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({ expected_version: version }),
  });
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).task;
}

export async function steerAgentTask(taskId: string, threadId: string, text: string, version: number): Promise<void> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/steer?${taskQuery(threadId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({ expected_version: version, text }),
  });
  if (!response.ok) throw new Error(await readApiError(response));
}

export async function getAgentTaskTodos(taskId: string, threadId: string): Promise<AgentTaskTodo[]> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/todos?${taskQuery(threadId)}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).todos;
}

export async function getAgentTaskRuns(taskId: string, threadId: string): Promise<AgentTaskRun[]> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/runs?${taskQuery(threadId)}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).runs;
}

export async function getAgentTaskTimeline(taskId: string, runId: string, threadId: string): Promise<{ task: AgentTaskSummary; run: AgentTaskRun; items: AgentTaskTimelineItem[] }> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId)}/timeline?${taskQuery(threadId)}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}

export async function getAgentTaskArtifacts(taskId: string, threadId: string, runId?: string): Promise<AgentTaskArtifact[]> {
  const query = new URLSearchParams({ thread_id: threadId });
  if (runId) query.set('run_id', runId);
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/artifacts?${query}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).artifacts;
}

export async function deleteAgentTask(taskId: string, threadId: string, version: number): Promise<void> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}?${taskQuery(threadId)}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json', 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({ expected_version: version }),
  });
  if (!response.ok) throw new Error(await readApiError(response));
}

export async function getAgentTaskSubagents(taskId: string, threadId: string): Promise<AgentTaskSubagentRun[]> {
  const response = await fetch(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/subagents?${taskQuery(threadId)}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return (await response.json()).subagents;
}

export async function downloadAgentTaskArtifact(taskId: string, artifactId: string, threadId: string): Promise<void> {
  const url = `${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(artifactId)}/download?${taskQuery(threadId)}`;
  window.open(url, '_blank', 'noopener,noreferrer');
}

export interface BuilderTestRuntimeInput {
  builder_session_id: string;
  base_workflow_id: string;
  spec: Record<string, any>;
  thread_id: string;
  question: string;
  llm_model: string;
  use_web_search?: boolean;
  use_reranker?: boolean;
  context_window?: number;
  replans?: number;
  allow_external_tools?: boolean;
  hitl_web_approval?: boolean;
  system_role_override?: string;
  tool_instructions_override?: Record<string, string>;
  custom_instructions_override?: string;
  client_timezone?: string;
  client_locale?: string;
  client_now_iso?: string;
  transient_messages?: Array<{ role: 'user' | 'assistant'; content: string }>;
  workflow_spec_fingerprint?: string;
}

export type BuilderTestStreamEnvelope = AgentExecutionStreamEnvelope;

export async function streamAgentWorkflowBuilderTest(
  payload: BuilderTestRuntimeInput,
  onEvent: (event: BuilderTestStreamEnvelope) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/internal/agent-workflows/test-runs/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  });
  await consumeAgentExecutionStream(response, onEvent);
}

export async function resumeAgentWorkflowBuilderTest(
  runId: string,
  payload: Omit<BuilderTestRuntimeInput, 'builder_session_id' | 'base_workflow_id' | 'spec' | 'question'> & {
    action: AgentRunResumeAction;
    interrupt_id: string;
    selected_option_ids?: string[];
    resume_token?: string;
    resume_version?: number;
  },
  onEvent: (event: BuilderTestStreamEnvelope) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/internal/agent-workflows/test-runs/${encodeURIComponent(runId)}/resume/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  });
  await consumeAgentExecutionStream(response, onEvent);
}

export async function cancelAgentWorkflowBuilderTest(runId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/internal/agent-workflows/test-runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error(await response.text());
}

export async function getLatestAgentWorkflowBuilderTest(
  builderSessionId: string,
  baseWorkflowId?: string,
): Promise<AgentRunDetails | null> {
  const params = new URLSearchParams({ builder_session_id: builderSessionId });
  if (baseWorkflowId) params.set('base_workflow_id', baseWorkflowId);
  const response = await fetch(`${API_BASE}/api/internal/agent-workflows/test-runs/latest?${params.toString()}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()).agent_run;
}

export interface ThreadChatResponse {
  answer: string;
  status?: string;
  user_message_id: string | null;
  assistant_message_id: string | null;
  used_chat_ids: string[];
  document_sources: { text: string; file_hash: string; score: number; source_type?: typeof ThreadFileSourceType.Pdf; title?: string; url?: string }[];
  web_sources?: WebSource[];
  reasoning?: string;
  reasoning_available?: boolean;
  reasoning_format?: ReasoningFormatValue;
  rewritten_query?: string;
  clarification_options?: string[] | null;
  agent_run_id?: string | null;
  agent_run_turn_kind?: string;
  agent_run_sequence?: number | null;
  agent_trace_refs?: AgentTraceRefs | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  agent_workflow_id?: string;
  route?: string;
  agent_route?: string;
  agent_route_reason?: string;
}

const threadChatPayload = (
  threadId: string,
  question: string,
  llmModel: string,
  useWebSearch: boolean = false,
  useReranker: boolean = true,
  contextWindowSize: number = 4096,
  replans?: number,
  systemRoleOverride?: string,
  toolInstructionsOverride?: Record<string, string>,
  customInstructionsOverride?: string,
  bypassClarification: boolean = false,
  hitlWebApproval?: boolean,
): Record<string, any> => {
  const payload: any = {
    thread_id: threadId,
    question,
    llm_model: llmModel,
    use_web_search: useWebSearch,
    use_reranker: useReranker,
    context_window: contextWindowSize,
    ...getBrowserRuntimeContext()
  };
  if (bypassClarification) {
    payload.bypass_clarification = true;
  }
  if (typeof hitlWebApproval === "boolean") {
    payload.hitl_web_approval = hitlWebApproval;
  }
  if (typeof replans === "number") {
    payload.replans = replans;
  }
  if (typeof systemRoleOverride === "string") {
    payload.system_role_override = systemRoleOverride;
  }
  if (toolInstructionsOverride && typeof toolInstructionsOverride === "object") {
    payload.tool_instructions_override = toolInstructionsOverride;
  }
  if (typeof customInstructionsOverride === "string") {
    payload.custom_instructions_override = customInstructionsOverride;
  }
  return payload;
};

export async function threadChat(
  threadId: string,
  question: string,
  llmModel: string,
  useWebSearch: boolean = false,
  useReranker: boolean = true,
  contextWindowSize: number = 4096,
  replans?: number,
  systemRoleOverride?: string,
  toolInstructionsOverride?: Record<string, string>,
  customInstructionsOverride?: string,
  bypassClarification: boolean = false,
  hitlWebApproval?: boolean,
): Promise<ThreadChatResponse> {
  const payload = threadChatPayload(threadId, question, llmModel, useWebSearch, useReranker, contextWindowSize, replans, systemRoleOverride, toolInstructionsOverride, customInstructionsOverride, bypassClarification, hitlWebApproval);
  const maxRetries = 2;
  let attempt = 0;

  while (true) {
    try {
      const res = await fetch(`${API_BASE}/api/threads/${threadId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errorText = await res.text();
        // Check if error is retryable (503 or transient)
        if (attempt < maxRetries && (res.status === 503 || res.status === 429)) {
          throw new Error(`RETRYABLE:${res.status}:${errorText}`);
        }
        throw new Error(errorText);
      }
      return res.json();
    } catch (err: any) {
      if (err.message?.startsWith('RETRYABLE:') || (attempt < maxRetries && (err.message?.includes('timeout') || err.message?.includes('Failed to fetch')))) {
        attempt++;
        const delay = attempt * 1000;
        console.log(`threadChat failed (attempt ${attempt}), retrying in ${delay}ms...`, err.message);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      throw err;
    }
  }
}

export async function streamThreadChat(
  threadId: string,
  question: string,
  llmModel: string,
  useWebSearch: boolean,
  useReranker: boolean,
  contextWindowSize: number,
  replans: number | undefined,
  systemRoleOverride: string | undefined,
  toolInstructionsOverride: Record<string, string> | undefined,
  customInstructionsOverride: string | undefined,
  bypassClarification: boolean,
  hitlWebApproval: boolean | undefined,
  onEvent: (event: AgentExecutionStreamEnvelope) => void,
  signal?: AbortSignal,
): Promise<void> {
  const payload = threadChatPayload(threadId, question, llmModel, useWebSearch, useReranker, contextWindowSize, replans, systemRoleOverride, toolInstructionsOverride, customInstructionsOverride, bypassClarification, hitlWebApproval);
  const response = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  });
  await consumeAgentExecutionStream(response, onEvent);
}

export async function getThreadIndexStatus(threadId: string): Promise<{
  thread_id: string;
  status: EmbeddingReadinessStatusValue;
  stats: any;
  embeddingModelReady?: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/indexing/status`);
  if (!res.ok) throw new Error(await res.text());
  const raw = await res.json();
  return {
    thread_id: raw.thread_id,
    status: raw.status,
    stats: raw.stats,
    embeddingModelReady: raw.embedding_model_ready,
  };
}
