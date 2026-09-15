import { API_BASE } from './api-config';
import type { CanvasSpec, ThreadCanvasRecord } from './canvas-spec';

async function readError(res: Response): Promise<string> {
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
}

export type CanvasCreateRequest = {
  spec: CanvasSpec;
  chat_turn_id?: string;
  supersedes_id?: string;
  idempotency_key?: string;
};

export async function listThreadCanvases(
  threadId: string,
  currentOnly = true,
): Promise<{ thread_id: string; canvases: ThreadCanvasRecord[] }> {
  const params = new URLSearchParams({ current_only: String(currentOnly) });
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/canvases?${params.toString()}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getThreadCanvas(threadId: string, canvasId: string): Promise<ThreadCanvasRecord> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/canvases/${canvasId}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function createThreadCanvas(
  threadId: string,
  request: CanvasCreateRequest,
): Promise<ThreadCanvasRecord> {
  const res = await fetch(`${API_BASE}/api/threads/${threadId}/canvases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}
