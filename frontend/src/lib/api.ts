const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const RAG_API_BASE = process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8001";

// ============ PDF Upload ============

export type IndexingStatus = 'pending' | 'indexing' | 'ready' | 'failed' | 'unknown';

export interface UploadResponse {
  sentences: any[];
  pdfUrl: string;
  fileHash: string;
  indexingStatus: IndexingStatus;
}

export interface FileIndexStatus {
  file_hash: string;
  status: IndexingStatus;
  error?: string;
  started_at?: string;
  finished_at?: string;
  progress?: number;
  message?: string;
}

export async function uploadPdf(file: File, embeddingModel: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("embedding_model", embeddingModel);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getFileIndexStatus(fileHash: string): Promise<FileIndexStatus> {
  const res = await fetch(`${API_BASE}/api/index-status/${fileHash}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface PdfData {
  sentences: any[];
  pdfUrl: string;
  fileHash: string;
}

export async function getPdfByHash(fileHash: string): Promise<PdfData> {
  const res = await fetch(`${API_BASE}/api/pdf/${fileHash}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ============ TTS ============

export async function ttsSentence(text: string, voice: string, speed: number) {
  const res = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ============ Thread API ============

export interface Thread {
  id: string;
  name: string;
  embed_model: string;
  created_at: string;
  message_count?: number;
  file_count?: number;
}

export interface ThreadFile {
  file_hash: string;
  file_name: string;
  file_path?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  isRecollected?: boolean;
}

export async function createThread(name: string, embedModel: string): Promise<Thread> {
  const res = await fetch(`${RAG_API_BASE}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, embed_model: embedModel })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listThreads(): Promise<{ threads: Thread[] }> {
  const res = await fetch(`${RAG_API_BASE}/threads`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThread(threadId: string): Promise<Thread & { files: ThreadFile[], stats: any }> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateThread(threadId: string, name: string): Promise<Thread> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}`, {
    method: "DELETE"
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function addFileToThread(
  threadId: string, 
  fileHash: string, 
  fileName: string, 
  text?: string
): Promise<any> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}/files`, {
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
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}/files`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThreadMessages(
  threadId: string, 
  limit: number = 100, 
  offset: number = 0
): Promise<{ messages: Message[] }> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}/messages?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteMessage(messageId: string): Promise<{ deleted_ids: string[] }> {
  const res = await fetch(`${RAG_API_BASE}/messages/${messageId}`, {
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
  contextWindowSize: number = 4096
): Promise<{
  answer: string;
  user_message_id: string;
  assistant_message_id: string;
  used_chat_ids: string[];
  pdf_sources: { text: string; file_hash: string; score: number }[];
}> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      question,
      llm_model: llmModel,
      use_web_search: useWebSearch,
      context_window: contextWindowSize
    })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getThreadIndexStatus(threadId: string): Promise<{
  thread_id: string;
  status: 'ready' | 'not_ready';
  stats: any;
}> {
  const res = await fetch(`${RAG_API_BASE}/threads/${threadId}/index-status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
