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
  pdfUrl: string;
  fileHash: string;
  fileName: string;
}

export async function uploadPdf(file: File, threadId: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
  pdfUrl: string;
  fileHash: string;
}

export async function getPdfByHash(fileHash: string, threadId: string): Promise<PdfData> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/files/${fileHash}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
  embed_model: string;
  settings?: ThreadSettings;
  thread_metadata?: ThreadMetadata;
  created_at: string;
  message_count?: number;
  file_count?: number;
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
  max_iterations: number;
  system_role: string;
  tool_instructions: Record<string, string>;
  custom_instructions: string;
  use_intent_agent: boolean;
  intent_agent_max_iterations: number;
  reasoning_mode: boolean;
  use_reranker: boolean;
}

export interface PromptToolDefinition {
  id: string;
  display_name: string;
  description: string;
  default_prompt: string;
}

export interface PromptDefaults {
  max_iterations: number;
  min_max_iterations: number;
  max_max_iterations: number;
  context_window: number;
  system_role: string;
  tool_instructions: Record<string, string>;
  custom_instructions: string;
  use_intent_agent: boolean;
  intent_agent_max_iterations: number;
  reasoning_mode: boolean;
  use_reranker?: boolean;
}

export interface ThreadFile {
  file_hash: string;
  file_name: string;
  file_path?: string;
  source_type?: 'pdf';
}

export interface WebSource {
  text: string;
  url: string;
  title: string;
  score?: number;
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
}

export async function createThread(name: string, embedModel: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, embed_model: embedModel })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listThreads(): Promise<{ threads: Thread[] }> {
  const res = await fetch(`${API_BASE}/api/threads`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThread(threadId: string): Promise<Thread & {
  files: ThreadFile[];
  stats: any;
  embed_model_ready?: boolean;
  stats_unavailable_reason?: string | null;
}> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateThread(threadId: string, name: string): Promise<Thread> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
  return res.json();
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
  intent_agent_ran?: boolean;
  reasoning_mode?: boolean;
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
  return res.json();
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
  file_hash: string;
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
  return res.json();
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

export async function threadChat(
  threadId: string,
  question: string,
  llmModel: string,
  useWebSearch: boolean = false,
  useReranker: boolean = true,
  contextWindowSize: number = 4096,
  maxIterations?: number,
  systemRoleOverride?: string,
  toolInstructionsOverride?: Record<string, string>,
  customInstructionsOverride?: string,
  useIntentAgent?: boolean,
  intentAgentMaxIterations?: number,
  reasoningMode?: boolean,
  intentAgentSkipClarify?: boolean
): Promise<{
  answer: string;
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
  if (typeof maxIterations === "number") {
    payload.max_iterations = maxIterations;
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
  if (typeof useIntentAgent === "boolean") {
    payload.use_intent_agent = useIntentAgent;
  }
  if (typeof intentAgentMaxIterations === "number") {
    payload.intent_agent_max_iterations = intentAgentMaxIterations;
  }
  if (typeof intentAgentSkipClarify === "boolean") {
    payload.intent_agent_skip_clarify = intentAgentSkipClarify;
  }
  if (typeof reasoningMode === "boolean") {
    payload.reasoning_mode = reasoningMode;
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
  embed_model_ready?: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/indexing/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
