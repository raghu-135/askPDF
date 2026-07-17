import {
  deserializeAnnotationItems,
  serializeAnnotationItems,
  type AnnotationTransferItem,
} from "./annotation-utils";
import { getBrowserRuntimeContext } from "./date-utils";

// Unified API base - RAG service handles all endpoints
const apiUrl = process.env.NEXT_PUBLIC_API_URL;
if (!apiUrl) {
  console.error("ERROR: NEXT_PUBLIC_API_URL environment variable is not set. Please configure it in docker-compose.yml");
}
export const API_BASE = apiUrl || "";

// ============ PDF Upload ============

export type ProcessStatus = 'pending' | 'running' | 'completed' | 'failed' | 'unknown';

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
  isCompleted: (status: ProcessStatus) => status === 'completed',
  isFailed: (status: ProcessStatus) => status === 'failed',
  isRunning: (status: ProcessStatus) => status === 'running',
  isPending: (status: ProcessStatus) => status === 'pending',
  isTerminal: (status: ProcessStatus) => status === 'completed' || status === 'failed',
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

export async function uploadPdf(file: File, threadId: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return mapUploadResponse(await res.json());
}

export async function getFileStatus(
  fileHash: string,
  threadId: string,
  options?: {
    section?: 'parsing' | 'indexing';
    embeddingModel?: string;
  }
): Promise<FileStatus | { parsing: ProcessSection } | { indexing: IndexingSection }> {
  const params = new URLSearchParams();
  if (options?.section) params.set("section", options.section);
  if (options?.embeddingModel) params.set("embedding_model", options.embeddingModel);
  const query = params.toString();
  const url = `${API_BASE}/api/threads/${threadId}/files/${fileHash}/status${query ? `?${query}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
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
    [key: string]: any;
  };
  auth_boundary?: {
    authoring_enabled?: boolean;
    custom_runtime_enabled?: boolean;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface AgentWorkflow {
  id: string;
  workflow_id?: string;
  name: string;
  description?: string;
  visibility?: string;
  is_builtin?: boolean;
  supports_replans?: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface AgentWorkflowValidationReport {
  valid: boolean;
  errors: string[];
  warnings: string[];
  schema_version?: number | null;
  [key: string]: any;
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
}

export interface ThreadAgentConfigValidationResponse {
  valid: boolean;
  workflow_id: string;
  validation: AgentWorkflowValidationReport;
  resolved_spec_json: AgentWorkflowBuilderSpec | Record<string, any>;
}

export interface AgentWorkflowGraphPreview {
  nodes?: Record<string, any>[];
  edges?: Record<string, any>[];
  executionPlan?: string[];
  selectedRoute?: string;
  [key: string]: any;
}

export interface ThreadAgentConfigPreviewResponse {
  thread_id?: string;
  workflow_id?: string;
  validation?: AgentWorkflowValidationReport;
  resolved_spec_json?: AgentWorkflowBuilderSpec | Record<string, any>;
  graph?: AgentWorkflowGraphPreview;
  prompt?: string;
  prompt_preview?: string;
  [key: string]: any;
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
  sourceType?: 'pdf' | 'browser';
}

interface RawThreadFile {
  file_hash: string;
  file_name: string;
  file_path?: string;
  source_type?: 'pdf' | 'browser';
}

const mapThreadFile = (raw: RawThreadFile): ThreadFile => ({
  fileHash: raw.file_hash,
  fileName: raw.file_name,
  filePath: raw.file_path,
  sourceType: raw.source_type,
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
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  isRecollected?: boolean;
  reasoning?: string;
  reasoning_available?: boolean;
  reasoning_format?: 'structured' | 'tagged_text' | 'none';
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
}

export type AgentRunResumeAction = 'approve' | 'approve_selected' | 'reject' | 'edit' | 'continue_without';

export interface AgentRunPendingInterrupt {
  interrupt_id: string;
  gate_id?: string | null;
  node_id?: string | null;
  type?: string | null;
  status?: 'pending' | 'resumed' | 'rejected' | 'expired' | string;
  requested_at?: string | null;
  expires_at?: string | null;
  default_action?: AgentRunResumeAction | string | null;
  allowed_actions?: AgentRunResumeAction[] | string[];
  mode?: 'approval' | 'choice' | 'review' | string | null;
  phase?: 'before' | 'after' | 'inside_tool' | string | null;
  selection_mode?: 'single' | 'multi' | 'single_or_multi' | string | null;
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
  resolved_spec_json?: Record<string, any>;
  metrics_json?: Record<string, any>;
  error_json?: Record<string, any> | null;
  started_at?: string;
  completed_at?: string | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  debug?: AgentRunDebug | null;
  turns?: {
    id: string;
    kind?: string | null;
    sequence?: number | null;
    trace_refs?: AgentTraceRefs | null;
  }[];
  [key: string]: any;
}

export interface AgentRunSummary {
  id: string;
  thread_id: string;
  workflow_id: string;
  status: string;
  started_at?: string | null;
  completed_at?: string | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  metrics?: Record<string, any>;
  error?: Record<string, any> | null;
  [key: string]: any;
}

export async function createThread(name: string, embeddingModel: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, embedding_model: embeddingModel })
  });
  if (!res.ok) throw new Error(await res.text());
  return mapThread(await res.json());
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
  options: { messageId?: string; name?: string } = {}
): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message_id: options.messageId,
      name: options.name,
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

export async function previewThreadAgentConfig(
  threadId: string,
  overrides: Record<string, any> = {}
): Promise<ThreadAgentConfigPreviewResponse> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/agent-config/preview`, {
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

export async function resumeAgentRun(
  runId: string,
  payload: {
    action: AgentRunResumeAction;
    interrupt_id: string;
    edited_payload?: Record<string, any>;
    client_metadata?: Record<string, any>;
    selected_option_ids?: string[];
    resume_token?: string;
    resume_version?: number;
    thread_id?: string;
  }
): Promise<{
  agent_run: AgentRunDetails;
  interrupt: AgentRunPendingInterrupt;
  outcome: string;
  duplicate: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/agent-runs/${runId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

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
  customInstructionsOverride?: string
): Promise<{
  answer: string;
  status?: string;
  user_message_id: string | null;
  assistant_message_id: string | null;
  used_chat_ids: string[];
  document_sources: { text: string; file_hash: string; score: number; source_type?: 'pdf'; title?: string; url?: string }[];
  web_sources?: WebSource[];
  reasoning?: string;
  reasoning_available?: boolean;
  reasoning_format?: 'structured' | 'tagged_text' | 'none';
  rewritten_query?: string;
  clarification_options?: string[] | null;
  agent_run_id?: string;
  agent_run_turn_kind?: string;
  agent_run_sequence?: number | null;
  agent_trace_refs?: AgentTraceRefs | null;
  pending_interrupt?: AgentRunPendingInterrupt | null;
  agent_workflow_id?: string;
  route?: string;
  agent_route?: string;
  agent_route_reason?: string;
}> {
  const payload: any = {
    thread_id: threadId,
    question,
    llm_model: llmModel,
    use_web_search: useWebSearch,
    use_reranker: useReranker,
    context_window: contextWindowSize,
    ...getBrowserRuntimeContext()
  };
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

export async function getThreadIndexStatus(threadId: string): Promise<{
  thread_id: string;
  status: 'ready' | 'not_ready' | 'blocked';
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
