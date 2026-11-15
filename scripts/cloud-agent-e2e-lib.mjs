/** Shared Node helpers for cloud-agent e2e scripts. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(__dirname, '..');

export const DEFAULT_ART_DIR = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
export const DEFAULT_BASE_URL = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
export const DEFAULT_API_URL = process.env.ASKPDF_E2E_API_URL || 'http://127.0.0.1:8000';

export function loadEnvAdmin() {
  const raw = fs.readFileSync(path.join(REPO_ROOT, '.env'), 'utf8');
  const admin = raw.match(/^ASKPDF_ADMIN_TOKEN=(.+)$/m)?.[1]?.trim();
  if (!admin) throw new Error('ASKPDF_ADMIN_TOKEN missing in .env');
  return admin;
}

export async function apiJson(url, { method = 'GET', body, admin } = {}) {
  const res = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${admin}`,
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${method} ${url} failed: ${res.status} ${text.slice(0, 400)}`);
  }
  return res.json();
}

export async function waitForFileReady(threadId, fileHash, admin, apiUrl = DEFAULT_API_URL) {
  for (let i = 0; i < 240; i += 1) {
    const status = await apiJson(`${apiUrl}/api/threads/${threadId}/files/${fileHash}/status`, { admin });
    const indexing = status?.indexing?.status;
    const projection = status?.document_processing?.projection?.status;
    if (indexing === 'completed' && projection === 'completed') return;
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`file ${fileHash} not indexed in time`);
}

export function defaultProjectId(projectsPayload) {
  const rows = Array.isArray(projectsPayload) ? projectsPayload : projectsPayload?.projects || [];
  return rows[0]?.id;
}